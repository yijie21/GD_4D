"""D4RTBridge — the S3 bridge wrapper around a frozen D4RT/OpenD4RT model.

It freezes the backbone, swaps every encoder self-attention block for a weight-exact
:class:`LoRAAttention` (so nothing changes until the LoRA adapters train), and adds the
imagined-time embedding. Two modes:

    bridge.set_mode("teacher")   # LoRA disabled → the frozen backbone, exactly
    bridge.set_mode("student")   # LoRA enabled  → the trainable, injected path

The distillation (S3) trains ``bridge.trainable_parameters()`` (LoRA A/B + imagined-time) so the
*student* on ``[obs … dream@t+j]`` reproduces the *teacher* on the full real clip.

NOTE (next increment): dream-token **gating** and wiring the imagined-time embedding into the encoder
forward are stubbed here (`set_dream_gate`, `self.imagined_time` exist but the student currently applies
its delta to all tokens). The identity milestone below does not depend on them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import LoRAAttention
from .imagined_time import ImaginedTimeEmbedding


class D4RTBridge(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.model = model
        for p in self.model.parameters():                 # freeze the backbone first
            p.requires_grad_(False)
        self.model.eval()

        enc = self.model.encoder
        self.attns: list[LoRAAttention] = []
        for blk in enc.blocks:
            blk.attn = LoRAAttention.from_mha(blk.attn, rank=rank, alpha=alpha)  # trainable A/B added
            self.attns.append(blk.attn)
        self.imagined_time = ImaginedTimeEmbedding(enc.hidden_dim)
        self.use_checkpoint = False                        # gradient checkpointing (set True for training)
        self.to(next(self.model.parameters()).device)     # new LoRA/imag-time modules → backbone device
        self.set_mode("student")

    def set_mode(self, mode: str) -> None:
        if mode not in ("teacher", "student"):
            raise ValueError("mode must be 'teacher' or 'student'")
        enabled = mode == "student"
        for la in self.attns:
            for m in (la.q, la.k, la.v, la.o):
                m.enabled = enabled
        self.mode = mode

    def set_dream_gate(self, gate: torch.Tensor | None) -> None:
        for la in self.attns:
            la.set_gate(gate)

    def trainable_parameters(self):
        for la in self.attns:
            yield from la.lora_parameters()
        yield from self.imagined_time.parameters()

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _sinusoidal(self):
        if getattr(self, "_sinu_fn", None) is None:
            from src.model.utils import sinusoidal_position_embedding  # vendored (on sys.path at runtime)
            self._sinu_fn = sinusoidal_position_embedding
        return self._sinu_fn

    def encode_video(self, video, aspect_ratio=None, dream_j=None):
        """Bridge-controlled re-implementation of the encoder forward (mirrors the vendored one).

        In **student** mode it (a) adds the imagined-time embedding to the dream tokens (the **last
        temporal patch** — the caller must place the dream there) and (b) **gates** every block's LoRA
        delta to those dream tokens, so observed frames stay bit-identical to the teacher. In **teacher**
        mode LoRA is disabled and no imagined-time is added → the frozen backbone, exactly (identity test).
        """
        enc = self.model.encoder
        extra = self.model._project_aspect_ratio_token(video=video, aspect_ratio=aspect_ratio)  # [B,1,C]|None
        x = video.permute(0, 2, 1, 3, 4)
        x = enc.patch_embed(x)
        x = enc._token_cap(x)
        b, c, tp, hp, wp = x.shape
        spatial, N = hp * wp, tp * hp * wp
        dream_start = (tp - 1) * spatial
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self._sinusoidal()(N, enc.hidden_dim, tokens.device).unsqueeze(0)

        student = self.mode == "student"
        if student and dream_j is not None:                # add imagined-time to the dream tokens only
            j = torch.as_tensor(dream_j, dtype=torch.float32, device=tokens.device).reshape(-1)
            if j.numel() == 1:
                j = j.expand(b)                            # scalar → per-sample [B]
            mask = torch.zeros(b, N, 1, device=tokens.device, dtype=tokens.dtype)
            mask[:, dream_start:, :] = 1.0
            tokens = tokens + mask * self.imagined_time(j).to(tokens.dtype)[:, None, :]

        if student:                                        # gate LoRA to the dream tokens
            g_vid = torch.zeros(b, N, 1, device=tokens.device, dtype=tokens.dtype)
            g_vid[:, dream_start:, :] = 1.0
            g_local = g_vid.reshape(b, tp, spatial, 1).reshape(b * tp, spatial, 1)
            g_global = (g_vid if extra is None else
                        torch.cat([g_vid, torch.zeros(b, extra.shape[1], 1, device=tokens.device, dtype=tokens.dtype)], 1))

        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        run = (lambda blk, t: checkpoint(blk, t, use_reentrant=False)) if ckpt else (lambda blk, t: blk(t))
        extra_tokens = extra
        for mode, block in zip(enc.block_modes, enc.blocks):
            block.attn.set_gate((g_local if mode == "local" else g_global) if student else None)
            if mode == "local":
                loc = tokens.reshape(b, tp, spatial, c).reshape(b * tp, spatial, c)
                loc = run(block, loc)
                tokens = loc.reshape(b, tp, spatial, c).reshape(b, N, c)
            elif extra_tokens is None:
                tokens = run(block, tokens)
            else:
                merged = run(block, torch.cat([tokens, extra_tokens], dim=1))
                tokens, extra_tokens = merged[:, :N], merged[:, N:]

        encoded = tokens if extra_tokens is None else torch.cat([tokens, extra_tokens], dim=1)
        return self.model.memory_proj(enc.final_norm(encoded))

    def decode_queries(self, video, query, memory):
        return self.model.decode_queries(video, query, memory)

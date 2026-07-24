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

    # delegate the queryable interface to the (now bridged) backbone
    def encode_video(self, video, aspect_ratio=None):
        return self.model.encode_video(video, aspect_ratio)

    def decode_queries(self, video, query, memory):
        return self.model.decode_queries(video, query, memory)

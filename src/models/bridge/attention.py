"""LoRA-wrapped self-attention that is a weight-exact replacement for ``nn.MultiheadAttention``.

The backbone's attention is packed inside ``nn.MultiheadAttention`` (one ``in_proj_weight`` for Q/K/V),
which LoRA can't target directly. ``LoRAAttention.from_mha`` unpacks it into explicit Q/K/V/O
``nn.Linear`` layers (copying the pretrained weights exactly) and wraps each in :class:`LoRALinear`.
With zero-init LoRA it reproduces the original MHA output to numerical precision (unit-tested), so the
frozen teacher is preserved; training the LoRA adapters is what turns it into the student.

Drop-in: it accepts the same ``(query, key, value, need_weights=...)`` call the backbone block makes and
returns ``(out, None)``. Self-attention only (query is used; key/value are ignored, as in the block).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear


class LoRAAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 q: nn.Linear, k: nn.Linear, v: nn.Linear, o: nn.Linear,
                 rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = LoRALinear(q, rank, alpha)
        self.k = LoRALinear(k, rank, alpha)
        self.v = LoRALinear(v, rank, alpha)
        self.o = LoRALinear(o, rank, alpha)

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention, rank: int = 16, alpha: float = 16.0) -> "LoRAAttention":
        E, nh = mha.embed_dim, mha.num_heads
        w = mha.in_proj_weight.detach()                   # [3E, E]
        b = mha.in_proj_bias.detach() if mha.in_proj_bias is not None else None

        def lin(wslice, bslice):
            m = nn.Linear(E, E, bias=b is not None)
            m.weight.data.copy_(wslice)
            if b is not None:
                m.bias.data.copy_(bslice)
            return m

        q = lin(w[:E], b[:E] if b is not None else None)
        k = lin(w[E:2 * E], b[E:2 * E] if b is not None else None)
        v = lin(w[2 * E:], b[2 * E:] if b is not None else None)
        o = nn.Linear(E, E, bias=mha.out_proj.bias is not None)
        o.weight.data.copy_(mha.out_proj.weight.detach())
        if mha.out_proj.bias is not None:
            o.bias.data.copy_(mha.out_proj.bias.detach())
        return cls(E, nh, q, k, v, o, rank, alpha)

    def set_gate(self, gate: torch.Tensor | None) -> None:
        for m in (self.q, self.k, self.v, self.o):
            m.gate = gate

    def forward(self, query, key=None, value=None, need_weights: bool = False, **kw):
        x = query                                          # [B, N, E] (self-attention)
        B, N, E = x.shape

        def heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, N, hd]

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        out = F.scaled_dot_product_attention(q, k, v)      # scale = 1/sqrt(hd), matches MHA
        out = out.transpose(1, 2).reshape(B, N, E)
        return self.o(out), None

    def lora_parameters(self):
        for m in (self.q, self.k, self.v, self.o):
            yield from m.lora_parameters()

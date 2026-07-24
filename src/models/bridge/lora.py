"""Gated LoRA linear for the S3 bridge.

A frozen base ``nn.Linear`` plus a low-rank delta ``B(A(x))·scale``. The up-projection ``B`` is
**zero-initialized**, so at init the delta is exactly 0 → the wrapped module reproduces the frozen
backbone bit-for-bit (verified by the M1 identity test). An optional per-token ``gate`` restricts the
delta to the dream-frame tokens (set by the bridge before a forward; ``None`` = apply to all tokens).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)                     # zero-init → delta == 0 at init
        self.gate: torch.Tensor | None = None             # [.., N, 1], set per-forward by the bridge
        self.enabled: bool = True                         # teacher = False (frozen base only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        delta = self.B(self.A(x)) * self.scale
        if self.gate is not None:
            delta = delta * self.gate.to(dtype=delta.dtype)
        return out + delta

    def lora_parameters(self):
        yield from self.A.parameters()
        yield from self.B.parameters()

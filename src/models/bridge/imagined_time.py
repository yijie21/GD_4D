"""Imagined-time embedding for the S3 bridge.

At inference the dream is placed at a nominal future time ``t+j`` (not the next frame). This module
maps the scalar offset ``j`` through sinusoidal features + a 2-layer MLP to a ``hidden_dim`` vector that
is added to the dream-frame tokens, telling the frozen backbone "this frame is j steps ahead."
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ImaginedTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, num_bands: int = 8) -> None:
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_bands, dtype=torch.float32) * math.pi)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_bands, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, j: torch.Tensor) -> torch.Tensor:
        """j: [B] (or scalar) imagined-time offset → [B, hidden_dim]."""
        j = torch.as_tensor(j, dtype=torch.float32, device=self.freqs.device).reshape(-1, 1)
        ang = j * self.freqs.view(1, -1)                   # [B, num_bands]
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(feats)

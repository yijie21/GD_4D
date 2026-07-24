"""S3 bridge — make an injected dream frame queryable by the frozen 4D backbone (Milestone M1)."""
from __future__ import annotations

from .attention import LoRAAttention
from .imagined_time import ImaginedTimeEmbedding
from .lora import LoRALinear
from .wrapper import D4RTBridge

__all__ = ["LoRALinear", "LoRAAttention", "ImaginedTimeEmbedding", "D4RTBridge"]

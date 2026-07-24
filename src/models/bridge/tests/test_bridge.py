"""Unit tests for the S3 bridge modules (fast, no backbone)."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.bridge import ImaginedTimeEmbedding, LoRAAttention, LoRALinear


def test_lora_zero_init_is_identity():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=4.0).eval()
    x = torch.randn(2, 5, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)       # zero-init B → delta == 0
    lora.enabled = False
    assert torch.allclose(lora(x), base(x), atol=1e-6)       # teacher: base only
    nn.init.normal_(lora.B.weight, std=0.2); lora.enabled = True
    assert not torch.allclose(lora(x), base(x), atol=1e-4)   # trained delta changes output


def test_lora_base_is_frozen_adapters_are_not():
    base = nn.Linear(6, 6)
    lora = LoRALinear(base, rank=2)
    assert not lora.base.weight.requires_grad and not lora.base.bias.requires_grad
    assert lora.A.weight.requires_grad and lora.B.weight.requires_grad


def test_lora_gate_restricts_delta_to_selected_tokens():
    base = nn.Linear(4, 4)
    lora = LoRALinear(base, rank=2, alpha=2.0)
    nn.init.normal_(lora.B.weight, std=0.5)
    x = torch.randn(1, 3, 4)
    lora.gate = torch.tensor([[[0.0], [1.0], [0.0]]])        # only token 1 gets the delta
    out, ref = lora(x), base(x)
    assert torch.allclose(out[:, 0], ref[:, 0], atol=1e-6)   # token 0 gated off
    assert not torch.allclose(out[:, 1], ref[:, 1], atol=1e-5)  # token 1 gated on


def test_lora_attention_matches_mha_exactly():
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(64, 4, batch_first=True).eval()
    x = torch.randn(2, 10, 64)
    ref, _ = mha(x, x, x, need_weights=False)
    la = LoRAAttention.from_mha(mha, rank=4, alpha=4.0).eval()  # zero-init → pure attention
    out, _ = la(x, x, x, need_weights=False)
    assert torch.allclose(ref, out, atol=1e-5)               # weight-exact conversion


def test_imagined_time_shape_and_finite():
    emb = ImaginedTimeEmbedding(32, num_bands=8)
    out = emb(torch.tensor([0.0, 5.0, 20.0]))
    assert out.shape == (3, 32) and torch.isfinite(out).all()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

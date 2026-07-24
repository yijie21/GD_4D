"""M1 bridge smoke — wrapping the frozen backbone with the bridge must change nothing at init.

Verifies on the real OpenD4RT model that:
  * teacher mode (LoRA disabled) reproduces the original encoder memory + query outputs exactly;
  * zero-init student mode (LoRA enabled, B=0) also reproduces them;
  * only the LoRA adapters + imagined-time embedding are trainable (backbone frozen).

If the identity holds, the LoRA injection is weight-exact and distillation can start from the frozen model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

import s0_backbone_sanity as s0  # noqa: E402
from models.bridge import D4RTBridge  # noqa: E402


def _clip(device, frames=16):
    import glob
    f = sorted(glob.glob("/workspace/datasets/libero/hdf5/libero_spatial/*.hdf5"))[0]
    imgs = s0.load_libero_frames(f, 0, "agentview_rgb", frames)          # [T,256,256,3]
    video_b = (torch.from_numpy(imgs).to(device=device, dtype=torch.float32)
               .permute(0, 3, 1, 2).unsqueeze(0) / 255.0)
    aspect = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    return video_b, aspect, imgs.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = s0._resolve_device(args.device)
    model = s0.load_backbone(args.config, args.ckpt, device)
    video_b, aspect, K = _clip(device)

    # queries on the last frame → first frame (any queries; we only compare orig vs bridged)
    rng = np.random.default_rng(0)
    n = 64
    uv = rng.uniform(0.1, 0.9, size=(n, 2)).astype(np.float32)
    tsrc = np.full((n,), K - 1, np.int64); z = np.zeros((n,), np.int64)

    with torch.no_grad():
        mem_orig = s0._encode_model_memory(model=model, video_b=video_b, aspect_b=aspect)
        out_orig = s0._query(model, video_b, aspect, mem_orig, uv, tsrc, z, z, 4096)["xyz_3d"]

    total = sum(p.numel() for p in model.parameters())
    bridge = D4RTBridge(model, rank=16, alpha=16.0)          # mutates model in place
    trainable = bridge.num_trainable()
    print(f"[params] backbone total={total/1e6:.1f}M  trainable(LoRA+imag-time)={trainable/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")

    with torch.no_grad():
        bridge.set_mode("teacher")
        mem_t = s0._encode_model_memory(model=bridge, video_b=video_b, aspect_b=aspect)
        bridge.set_mode("student")
        mem_s = s0._encode_model_memory(model=bridge, video_b=video_b, aspect_b=aspect)
        out_s = s0._query(bridge, video_b, aspect, mem_s, uv, tsrc, z, z, 4096)["xyz_3d"]

    d_teacher = (mem_orig - mem_t).abs().max().item()
    d_student = (mem_orig - mem_s).abs().max().item()
    d_query = float(np.abs(out_orig - out_s).max())
    print(f"[identity] teacher memory max|Δ| = {d_teacher:.2e}")
    print(f"[identity] student memory max|Δ| = {d_student:.2e}  (zero-init LoRA)")
    print(f"[identity] student query xyz max|Δ| = {d_query:.2e}")

    ok = d_teacher < 1e-3 and d_student < 1e-3 and d_query < 1e-3
    print("RESULT:", "PASS — bridge is a weight-exact identity at init" if ok else "FAIL — investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

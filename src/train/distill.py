"""S3 bridge distillation — full training path: bidirectional queries, all offsets, held-out recall.

Train the LoRA + imagined-time bridge so the STUDENT (obs + dream@t+j) reproduces the TEACHER (full
real clip) correspondences, for dreams at multiple horizons j and in BOTH directions (o_t↔dream).
GPU0, teacher targets cached.

  * TEACHER (frozen): full clip [o_{t-h+1..t}, f_{t+1..t+n}] (28f). For each offset j, query o_t↔f_{t+j}
    (forward o_t→f_{t+j}, backward f_{t+j}→o_t), ref0 coords. Frozen → cached.
  * STUDENT (bridge): [obs×h, f_{t+j}, f_{t+j}] (10f; dream duplicated → clean last temporal patch),
    imagined-time j on the dream, LoRA gated to it. Same bidirectional queries.
  * Loss = Σ_dir SmoothL1(xyz) + 0.1·MSE(conf).
  * Acceptance metric: uv-reprojection **recall@3px** of student vs teacher on a held-out split
    (plan gate: ≥95% of the real-frame level; the teacher IS the real-frame reference).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC)); sys.path.insert(0, str(_SRC / "eval"))

import s0_backbone_sanity as s0  # noqa: E402
from data import WindowConfig, build_tuple, plan_tuples  # noqa: E402
from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402
from models.bridge import D4RTBridge  # noqa: E402

O_T, DREAM_S, T_CAM = 7, 8, 0        # o_t frame idx; student dream frame idx; ref frame
PX = 255.0                           # 256px → pixel scale
GRID_N = 24


def _stack(frames, size=256):
    return np.stack([cv2.resize(np.ascontiguousarray(f), (size, size), interpolation=cv2.INTER_LINEAR) for f in frames])


def teacher_clip(tup):
    return _stack(np.concatenate([tup.obs_frames, tup.waypoint_frames], 0))       # [28,256,256,3]


def student_clip(tup, j):
    dream = tup.waypoint_frames[j - 1]                                            # f_{t+j}
    return _stack(np.concatenate([tup.obs_frames, dream[None], dream[None]], 0))  # [10,256,256,3]


def to_video(arr_u8, device):
    return torch.from_numpy(arr_u8).to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0


def grid_uv():
    xs = np.linspace(0.08, 0.92, GRID_N, dtype=np.float32)
    return np.stack(np.meshgrid(xs, xs, indexing="xy"), -1).reshape(-1, 2)


def decode(bridge, video_b, mem, uv, t_src, t_tgt):
    B, M, dev = video_b.shape[0], uv.shape[0], video_b.device
    u = torch.as_tensor(uv[:, 0], dtype=torch.float32, device=dev).view(1, M).expand(B, M).contiguous()
    v = torch.as_tensor(uv[:, 1], dtype=torch.float32, device=dev).view(1, M).expand(B, M).contiguous()
    q = {"u": u, "v": v,
         "t_src": torch.as_tensor(t_src, device=dev).view(1, 1).expand(B, M).contiguous().long() if np.isscalar(t_src)
                  else torch.as_tensor(t_src, device=dev).view(B, 1).expand(B, M).contiguous().long(),
         "t_tgt": torch.as_tensor(t_tgt, device=dev).view(1, 1).expand(B, M).contiguous().long() if np.isscalar(t_tgt)
                  else torch.as_tensor(t_tgt, device=dev).view(B, 1).expand(B, M).contiguous().long(),
         "t_cam": torch.full((B, M), T_CAM, dtype=torch.long, device=dev)}
    pred = bridge.decode_queries(video_b, q, mem)
    return pred["xyz_3d"], pred["confidence"], pred["uv_2d"]


def gather_tuples(n, cfg):
    out = []
    for f in sorted((LIBERO_ROOT / "libero_spatial").glob("*.hdf5")):
        for di in range(6):                                  # a few demos per task file
            try:
                traj = LiberoTrajectory(f, di)
            except IndexError:
                break
            for sp in plan_tuples(traj.episode_id, len(traj), cfg)[:4]:
                out.append(build_tuple(traj, sp))
                if len(out) >= n:
                    return out
    return out


def build_samples(tuples, offsets, device, uv, bridge, batch):
    """Cache teacher targets + student clips for every (tuple, offset). Returns a dict of arrays."""
    S = len(tuples) * len(offsets)
    stud = np.zeros((S, 10, 256, 256, 3), np.uint8)
    jvals = np.zeros((S,), np.float32)
    tf_xyz = torch.zeros(S, uv.shape[0], 3); tf_uv = torch.zeros(S, uv.shape[0], 2); tf_c = torch.zeros(S, uv.shape[0])
    tb_xyz = torch.zeros_like(tf_xyz); tb_uv = torch.zeros_like(tf_uv); tb_c = torch.zeros_like(tf_c)
    s = 0
    idx_of = {}
    for ti, tup in enumerate(tuples):
        for j in offsets:
            stud[s] = student_clip(tup, j); jvals[s] = j; idx_of[(ti, j)] = s; s += 1
    bridge.set_mode("teacher")
    with torch.no_grad():
        for i in range(0, len(tuples), batch):
            tb = tuples[i:i + batch]
            tv = to_video(np.stack([teacher_clip(t) for t in tb]), device)
            asp = torch.ones((tv.shape[0], 1), device=device)
            mem = bridge.encode_video(tv, asp)
            for j in offsets:
                xf, cf, uf = decode(bridge, tv, mem, uv, O_T, O_T + j)          # o_t → f_{t+j}
                xb, cb, ub = decode(bridge, tv, mem, uv, O_T + j, O_T)          # f_{t+j} → o_t
                for k, tup in enumerate(tb):
                    sidx = idx_of[(i + k, j)]
                    tf_xyz[sidx], tf_c[sidx], tf_uv[sidx] = xf[k].cpu(), cf[k].cpu(), uf[k].cpu()
                    tb_xyz[sidx], tb_c[sidx], tb_uv[sidx] = xb[k].cpu(), cb[k].cpu(), ub[k].cpu()
    return dict(stud=stud, j=jvals, tf_xyz=tf_xyz, tf_uv=tf_uv, tf_c=tf_c, tb_xyz=tb_xyz, tb_uv=tb_uv, tb_c=tb_c)


def student_outputs(bridge, D, idx, device, uv):
    sv = to_video(D["stud"][idx], device)
    asp = torch.ones((sv.shape[0], 1), device=device)
    jb = torch.as_tensor(D["j"][idx], device=device)
    mem = bridge.encode_video(sv, asp, dream_j=jb)
    xf, cf, uf = decode(bridge, sv, mem, uv, O_T, DREAM_S)                       # o_t → dream
    xb, cb, ub = decode(bridge, sv, mem, uv, DREAM_S, O_T)                       # dream → o_t
    return (xf, cf, uf), (xb, cb, ub)


def recall_at_3px(bridge, D, device, uv, batch=8):
    bridge.set_mode("student")
    hits_f = hits_b = tot = 0
    with torch.no_grad():
        for i in range(0, len(D["j"]), batch):
            idx = np.arange(i, min(i + batch, len(D["j"])))
            (xf, cf, uf), (xb, cb, ub) = student_outputs(bridge, D, idx, device, uv)
            ef = (uf.cpu() - D["tf_uv"][idx]).norm(dim=-1) * PX
            eb = (ub.cpu() - D["tb_uv"][idx]).norm(dim=-1) * PX
            hits_f += (ef < 3).float().sum().item(); hits_b += (eb < 3).float().sum().item()
            tot += ef.numel()
    return hits_f / tot, hits_b / tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-train", type=int, default=40); ap.add_argument("--n-heldout", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_SRC.parents[0] / "experiments" / "M1_bridge_distill"))
    args = ap.parse_args()

    device = torch.device(args.device)
    model = s0.load_backbone(args.config, args.ckpt, device)
    bridge = D4RTBridge(model, rank=16, alpha=16.0)
    bridge.use_checkpoint = True                            # gradient checkpointing → fits big backbone
    cfg = WindowConfig()
    uv = grid_uv()
    offsets = sorted(set(max(1, round(f * cfg.n)) for f in (0.25, 0.5, 0.75, 1.0)))   # dream horizons j

    tuples = gather_tuples(args.n_train + args.n_heldout, cfg)
    tr, ho = tuples[:args.n_train], tuples[args.n_train:args.n_train + args.n_heldout]
    print(f"tuples: train={len(tr)} heldout={len(ho)}  offsets(j)={offsets}  queries={uv.shape[0]}")
    Dtr = build_samples(tr, offsets, device, uv, bridge, args.batch)
    Dho = build_samples(ho, offsets, device, uv, bridge, args.batch)
    Str = len(Dtr["j"])
    print(f"samples: train={Str}  heldout={len(Dho['j'])}  (teacher targets cached)")

    rf0, rb0 = recall_at_3px(bridge, Dho, device, uv)        # untrained baseline
    print(f"held-out recall@3px BEFORE: fwd {rf0:.3f}  bwd {rb0:.3f}")

    bridge.set_mode("student")
    opt = torch.optim.Adam(list(bridge.trainable_parameters()), lr=args.lr)
    losses = []
    for step in range(args.steps):
        idx = np.random.default_rng(step).choice(Str, size=min(args.batch, Str), replace=False)
        (xf, cf, uf), (xb, cb, ub) = student_outputs(bridge, Dtr, idx, device, uv)
        loss = (F.smooth_l1_loss(xf, Dtr["tf_xyz"][idx].to(device)) + 0.1 * F.mse_loss(cf, Dtr["tf_c"][idx].to(device))
                + F.smooth_l1_loss(xb, Dtr["tb_xyz"][idx].to(device)) + 0.1 * F.mse_loss(cb, Dtr["tb_c"][idx].to(device)))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    rf1, rb1 = recall_at_3px(bridge, Dho, device, uv)
    print(f"held-out recall@3px AFTER:  fwd {rf1:.3f}  bwd {rb1:.3f}   (gate ⛔ ≥0.95)")
    print(f"RESULT: loss {losses[0]:.4f}->{losses[-1]:.4f}; recall fwd {rf0:.2f}->{rf1:.2f} bwd {rb0:.2f}->{rb1:.2f}")

    out = Path(args.out); (out / "viz").mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    a1.plot(losses, lw=1.4); a1.set_xlabel("step"); a1.set_ylabel("distill loss")
    a1.set_title("bidirectional distill loss (all offsets)")
    a2.bar(["fwd\nbefore", "fwd\nafter", "bwd\nbefore", "bwd\nafter"], [rf0, rf1, rb0, rb1],
           color=["#bbb", "#4c78a8", "#bbb", "#54a24b"])
    a2.axhline(0.95, color="crimson", ls="--", lw=1, label="gate 0.95"); a2.set_ylim(0, 1)
    a2.set_ylabel("held-out recall@3px"); a2.set_title("student→teacher correspondence recall"); a2.legend()
    fig.tight_layout(); fig.savefig(out / "viz" / "distill_recall.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", out / "viz" / "distill_recall.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

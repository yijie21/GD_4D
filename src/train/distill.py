"""S3 bridge distillation — train the student (obs + dream) to match the teacher (full real clip).

First increment of the training loop (GPU0, teacher-cached). For each tuple and horizon j=n (goal):
  * TEACHER  = frozen backbone on the full real clip [o_{t-h+1..t}, f_{t+1..t+n}] (28 frames);
               query the 3D position (ref0 coords) of a grid on the current frame o_t at the goal time.
               Frozen → cache these targets once.
  * STUDENT  = bridge (LoRA on) on [obs×h, dream, dream] (10 frames; dream duplicated so it is the clean
               last temporal patch and isn't dropped by the 2-frame temporal patching); same query.
  * Loss     = SmoothL1(xyz) + w·MSE(confidence), student → teacher.

NOTE (next increments): imagined-time wiring into the encoder forward + dream-token gating are not yet
active here (student LoRA applies to all tokens; no j-embedding) — this increment validates the training
machinery (grads flow to LoRA, loss decreases). See plan §M1 tasks.
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
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "eval"))                  # for `import s0_backbone_sanity`

import s0_backbone_sanity as s0  # noqa: E402
from data import WindowConfig, build_tuple, plan_tuples  # noqa: E402
from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402
from models.bridge import D4RTBridge  # noqa: E402

T_SRC, TGT_TEACHER, TGT_STUDENT, T_CAM = 7, 27, 8, 0     # o_t frame, goal(teacher/student), ref frame


def _resize_stack(frames, size=256):
    return np.stack([cv2.resize(np.ascontiguousarray(f), (size, size), interpolation=cv2.INTER_LINEAR) for f in frames])


def clips_for(tup):
    teacher = _resize_stack(np.concatenate([tup.obs_frames, tup.waypoint_frames], 0))      # [28,256,256,3]
    goal = tup.goal_frame
    student = _resize_stack(np.concatenate([tup.obs_frames, goal[None], goal[None]], 0))    # [10,256,256,3]
    return teacher, student


def to_video(arr_u8, device):
    return (torch.from_numpy(arr_u8).to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0)


def grid_uv(n_side=24, m=0.08):
    xs = np.linspace(m, 1 - m, n_side, dtype=np.float32)
    return np.stack(np.meshgrid(xs, xs, indexing="xy"), -1).reshape(-1, 2)


def query(bridge, video_b, aspect_b, uv, t_src, t_tgt, dream_j=None):
    B, M, dev = video_b.shape[0], uv.shape[0], video_b.device
    u = torch.as_tensor(uv[:, 0], dtype=torch.float32, device=dev).view(1, M).expand(B, M).contiguous()
    v = torch.as_tensor(uv[:, 1], dtype=torch.float32, device=dev).view(1, M).expand(B, M).contiguous()
    q = {"u": u, "v": v,
         "t_src": torch.full((B, M), t_src, dtype=torch.long, device=dev),
         "t_tgt": torch.full((B, M), t_tgt, dtype=torch.long, device=dev),
         "t_cam": torch.full((B, M), T_CAM, dtype=torch.long, device=dev)}
    mem = bridge.encode_video(video_b, aspect_b, dream_j=dream_j)   # imagined-time on dream tokens (student)
    pred = bridge.decode_queries(video_b, q, mem)
    return pred["xyz_3d"], pred["confidence"]               # [B,M,3], [B,M]


def gather_tuples(n, cfg):
    files = sorted((LIBERO_ROOT / "libero_spatial").glob("*.hdf5"))
    out = []
    for f in files:
        traj = LiberoTrajectory(f, 0)
        for sp in plan_tuples(traj.episode_id, len(traj), cfg)[:2]:
            out.append(build_tuple(traj, sp))
            if len(out) >= n:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-tuples", type=int, default=8); ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_SRC.parents[0] / "experiments" / "M1_bridge_distill"))
    args = ap.parse_args()

    device = torch.device(args.device)
    model = s0.load_backbone(args.config, args.ckpt, device)
    bridge = D4RTBridge(model, rank=16, alpha=16.0)
    cfg = WindowConfig()
    uv = grid_uv()

    tuples = gather_tuples(args.n_tuples, cfg)
    teach_np = np.stack([clips_for(t)[0] for t in tuples])   # [N,28,256,256,3]
    stud_np = np.stack([clips_for(t)[1] for t in tuples])    # [N,10,256,256,3]
    N = len(tuples)
    aspect = lambda b: torch.ones((b, 1), dtype=torch.float32, device=device)
    print(f"tuples={N}  student_clip={stud_np.shape[1]}f  teacher_clip={teach_np.shape[1]}f  queries={uv.shape[0]}")

    # --- cache teacher targets (frozen) ---
    bridge.set_mode("teacher")
    xyz_t, conf_t = [], []
    with torch.no_grad():
        for i in range(0, N, args.batch):
            vb = to_video(teach_np[i:i + args.batch], device)
            x, c = query(bridge, vb, aspect(vb.shape[0]), uv, T_SRC, TGT_TEACHER)
            xyz_t.append(x.float().cpu()); conf_t.append(c.float().cpu())
    xyz_t = torch.cat(xyz_t); conf_t = torch.cat(conf_t)      # [N,M,3], [N,M]
    print(f"teacher cached: xyz {tuple(xyz_t.shape)}  (ref0 3D targets)")

    # --- train student (LoRA + imagined-time; delta gated to dream tokens) ---
    bridge.set_mode("student")
    params = list(bridge.trainable_parameters())            # LoRA A/B + imagined-time MLP
    opt = torch.optim.Adam(params, lr=args.lr)
    losses = []
    for step in range(args.steps):
        idx = np.random.default_rng(step).choice(N, size=min(args.batch, N), replace=False)
        vb = to_video(stud_np[idx], device)
        x_s, c_s = query(bridge, vb, aspect(vb.shape[0]), uv, T_SRC, TGT_STUDENT, dream_j=float(cfg.n))
        tgt_x = xyz_t[idx].to(device); tgt_c = conf_t[idx].to(device)
        loss = F.smooth_l1_loss(x_s, tgt_x) + 0.1 * F.mse_loss(c_s, tgt_c)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    g = sum(p.grad.abs().sum().item() for p in params if p.grad is not None)
    print(f"RESULT: loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({100*(1-losses[-1]/losses[0]):.0f}% down)  |grad|>0: {g>0}")

    out = Path(args.out); (out / "viz").mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axp = plt.subplots(figsize=(6, 4), dpi=120)
    axp.plot(losses, lw=1.5); axp.set_xlabel("step"); axp.set_ylabel("distill loss")
    axp.set_title(f"S3 bridge distillation (overfit {N} tuples) — student → teacher")
    fig.tight_layout(); fig.savefig(out / "viz" / "distill_loss.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", out / "viz" / "distill_loss.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

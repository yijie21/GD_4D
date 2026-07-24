"""M3 cross-consistency de-risk — can a REFERENCE-FREE signal flag a wrong goal?

M1 (premise) + M2 proved the frozen backbone's *self*-consistency (cycle/visibility
of the goal alone) is blind to dream correctness. This tests the alternative: check
the goal against the OBSERVED CONTEXT (obs window), never against itself or a clean
reference.

Setup: an obs window spaced at the goal horizon H — frames at times [t-2H, t-H, t]
(clip idx 0,1,2) + a candidate goal (clip idx 3). One clip-step ~= H real frames, so
the goal sits exactly one velocity-step after o_t and motion extrapolation is well
defined. All queries use the frozen backbone (`decode_queries`), obs window is REAL.

Candidate goals per window:
  * correct    : real frame[t+H] of THIS demo                 (label 0)
  * wrong_demo : real frame[~t+H] of a DIFFERENT demo, same task (label 1) -- hard: same scene
  * wrong_task : real frame of a DIFFERENT task               (label 1) -- easy: wrong scene

Reference-free signals (all from [obs window, goal] only):
  * S_motion   : moving points -- ||goal_pos - (o_t + velocity)||  (does the goal continue the motion?)
  * S_bg       : static points -- ||goal_pos - o_t||             (did the scene move?)
  * S_incomplete: 1 - min(fwd, rev) grid correspondence completeness (hallucinated/dropped content)
  * S_selfcyc  : BASELINE self-consistency cycle error goal->o_t->goal (expected BLIND, per M1/M2)

Read-out: per-signal AUROC separating correct from each wrong type. If S_motion
separates wrong_demo (same scene, wrong trajectory) while S_selfcyc does not, a
reference-free cross-consistency detector is viable.

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    CUDA_VISIBLE_DEVICES=0 python experiments/M3_cross_consistency/derisk.py \
        --config checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml \
        --ckpt   checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC = Path("/workspace/code/GD_4D/src")
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "eval"))
import s0_backbone_sanity as s0  # noqa: E402
from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402

SIZE = 256
PXHW = np.array([SIZE - 1, SIZE - 1], np.float32)


def _resize(f):
    return cv2.resize(np.ascontiguousarray(f), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def _block_shuffle(img, block=32, seed=0):
    """Permute block positions: identical local content, destroyed global geometry."""
    rng = np.random.default_rng(seed)
    n = SIZE // block
    blocks = [img[i * block:(i + 1) * block, j * block:(j + 1) * block]
              for i in range(n) for j in range(n)]
    order = rng.permutation(len(blocks))
    out = np.empty_like(img)
    for idx, p in enumerate(order):
        i, j = divmod(idx, n)
        out[i * block:(i + 1) * block, j * block:(j + 1) * block] = blocks[p]
    return out


def encode(model, clip_u8):
    dev = next(model.parameters()).device
    video_b = (torch.from_numpy(clip_u8).to(dev, torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0)
    aspect_b = torch.tensor([[1.0]], dtype=torch.float32, device=dev)
    mem = s0._encode_model_memory(model=model, video_b=video_b, aspect_b=aspect_b)
    return video_b, aspect_b, mem


def _grid(g=24, m=0.04):
    xs = np.linspace(m, 1 - m, g, np.float32)
    ys = np.linspace(m, 1 - m, g, np.float32)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)


def q(model, vb, ab, mem, uv, s, t, chunk=4096):
    n = uv.shape[0]
    fs = np.full((n,), s, np.int64)
    ft = np.full((n,), t, np.int64)
    z = np.zeros((n,), np.int64)
    r = s0._query(model, vb, ab, mem, uv, fs, ft, z, chunk)
    return r["uv_2d"], s0._sigmoid(r["visibility"])


def signals(model, clip_u8):
    """Return dict of reference-free signals for clip [o0,o1,o2(=o_t), goal(idx3)]."""
    vb, ab, mem = encode(model, clip_u8)
    OT, GOAL = 2, 3
    uv2 = _grid(24)                                    # points on o_t
    # motion over the obs window (o_t -> o_{t-H}) in px
    uv1, vis1 = q(model, vb, ab, mem, uv2, OT, 1)      # o_t -> o_{t-H}
    uv0, vis0 = q(model, vb, ab, mem, uv2, OT, 0)      # o_t -> o_{t-2H}
    disp_win = np.linalg.norm((uv0 - uv2) * PXHW, axis=1)
    vel = (uv2 - uv1)                                  # per clip-step velocity (normalized)
    # goal correspondence
    uvg, visg = q(model, vb, ab, mem, uv2, OT, GOAL)   # o_t -> goal
    vis_ok = (vis1 > 0.5) & (visg > 0.5)

    moving = (disp_win > 8.0) & vis_ok
    static = (disp_win < 3.0) & vis_ok

    out = {}
    # S_motion: does the goal continue the observed motion?
    if moving.sum() >= 5:
        pred = uv2[moving] + vel[moving]
        res = np.linalg.norm((uvg[moving] - pred) * PXHW, axis=1)
        out["S_motion"] = float(np.median(res))
    else:
        out["S_motion"] = np.nan
    # S_bg: did the static scene move in the goal?
    if static.sum() >= 5:
        res = np.linalg.norm((uvg[static] - uv2[static]) * PXHW, axis=1)
        out["S_bg"] = float(np.median(res))
    else:
        out["S_bg"] = np.nan
    # S_incomplete: grid correspondence completeness both ways
    _, visf = q(model, vb, ab, mem, _grid(24), OT, GOAL)     # o_t -> goal
    uvr, visr = q(model, vb, ab, mem, _grid(24), GOAL, OT)    # goal -> o_t
    inb = (uvr >= 0).all(1) & (uvr <= 1).all(1)
    fwd = float((visf > 0.5).mean())
    rev = float(((visr > 0.5) & inb).mean())
    out["S_incomplete"] = 1.0 - min(fwd, rev)
    # S_selfcyc baseline: goal -> o_t -> goal
    uvc = _grid(24)
    uvto, _ = q(model, vb, ab, mem, uvc, GOAL, OT)
    uvback, _ = q(model, vb, ab, mem, uvto, OT, GOAL)
    out["S_selfcyc"] = float(np.median(np.linalg.norm((uvback - uvc) * PXHW, axis=1)))
    return out


def build_windows(tasks_files, H=20, per_task=3, wins_per_demo=1):
    """Yield dicts with obs clip frames + candidate goals (correct/wrong_demo/wrong_task)."""
    trajs = {f.stem: [LiberoTrajectory(f, di) for di in range(6)] for f in tasks_files}
    items = []
    task_names = list(trajs.keys())
    for ti, f in enumerate(tasks_files):
        name = f.stem
        other_task = task_names[(ti + 1) % len(task_names)]
        for di in range(per_task):
            tr = trajs[name][di]
            T = len(tr)
            if T < 3 * H + 1:
                continue
            for w in range(wins_per_demo):
                t = 2 * H + (T - 1 - 3 * H) // 2 + w * H
                if t + H > T - 1 or t - 2 * H < 0:
                    continue
                obs = [_resize(tr.frame(t - 2 * H)), _resize(tr.frame(t - H)), _resize(tr.frame(t))]
                g_correct = _resize(tr.frame(t + H))
                # wrong_demo: another demo, same task, ~same relative goal time
                od = (di + 3) % 6
                trd = trajs[name][od]
                gd_t = min(t + H, len(trd) - 1)
                g_wrong_demo = _resize(trd.frame(gd_t))
                # wrong_task: a demo of a different task
                trt = trajs[other_task][di]
                g_wrong_task = _resize(trt.frame(min(t + H, len(trt) - 1)))
                # DIAGNOSTIC extreme negatives: same content, destroyed geometry / no content
                g_shuffled = _block_shuffle(g_correct, block=32)
                g_blank = np.full_like(g_correct, 127)
                items.append({"task": name, "demo": di, "t": t,
                              "obs": obs,
                              "goals": {"correct": g_correct, "wrong_demo": g_wrong_demo,
                                        "wrong_task": g_wrong_task,
                                        "shuffled": g_shuffled, "blank": g_blank}})
    return items


def auroc(score, label):
    from s4_disagreement_derisk import auroc as _a
    return _a(np.asarray(score, float), np.asarray(label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--per-task", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    device = s0._resolve_device(args.device)
    model = s0.load_backbone(args.config, args.ckpt, device)

    files = sorted((LIBERO_ROOT / "libero_spatial").glob("*.hdf5"))[:5]
    items = build_windows(files, H=args.H, per_task=args.per_task)
    print(f"windows: {len(items)}  (H={args.H})")

    SIGS = ["S_motion", "S_bg", "S_incomplete", "S_selfcyc"]
    rows = []  # (variant, {signals})
    for k, it in enumerate(items):
        for variant, gimg in it["goals"].items():
            clip = np.stack(it["obs"] + [gimg])
            s = signals(model, clip)
            s["variant"] = variant
            rows.append(s)
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{len(items)} windows done", flush=True)

    def col(variant, sig):
        return np.array([r[sig] for r in rows if r["variant"] == variant and not np.isnan(r[sig])])

    report = {"n_windows": len(items), "H": args.H, "auroc": {}}
    for sig in SIGS:
        corr = col("correct", sig)
        for neg in ("wrong_demo", "wrong_task", "shuffled", "blank"):
            wr = col(neg, sig)
            sc = np.concatenate([corr, wr])
            lab = np.concatenate([np.zeros(len(corr)), np.ones(len(wr))])
            report["auroc"][f"{sig}__vs__{neg}"] = round(auroc(sc, lab), 3)
        report.setdefault("medians", {})[sig] = {
            v: (round(float(np.median(col(v, sig))), 2) if len(col(v, sig)) else None)
            for v in ("correct", "wrong_demo", "wrong_task", "shuffled", "blank")
        }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "derisk_report.json").write_text(json.dumps(report, indent=2))
    _plot(rows, SIGS, out / "derisk_distributions.png")
    print("\n[report]")
    for sig in SIGS:
        m = report["medians"][sig]
        print(f"  {sig:14s} median  correct={m['correct']}  w_demo={m['wrong_demo']}  "
              f"w_task={m['wrong_task']}  shuffled={m['shuffled']}  blank={m['blank']}")
        a = report["auroc"]
        print(f"                 AUROC  w_demo={a[f'{sig}__vs__wrong_demo']}  w_task={a[f'{sig}__vs__wrong_task']}  "
              f"shuffled={a[f'{sig}__vs__shuffled']}  blank={a[f'{sig}__vs__blank']}")
    print(f"[save] {out/'derisk_report.json'} + derisk_distributions.png")


def _plot(rows, SIGS, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SIGS), figsize=(4 * len(SIGS), 3.6), dpi=120)
    cols = {"correct": "#4c78a8", "wrong_demo": "#e45756", "wrong_task": "#f58518"}
    for ax, sig in zip(axes, SIGS):
        for variant, c in cols.items():
            v = np.array([r[sig] for r in rows if r["variant"] == variant and not np.isnan(r[sig])])
            if len(v):
                ax.hist(v, bins=15, alpha=0.55, color=c, density=True, label=variant)
        ax.set_title(sig, fontsize=10); ax.set_xlabel("px" if sig != "S_incomplete" else "1-completeness")
        ax.legend(fontsize=7)
    fig.suptitle("M3 cross-consistency de-risk — reference-free signals: correct vs wrong goals", fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()

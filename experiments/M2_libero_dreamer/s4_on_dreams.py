"""Does the S4 geometric disagreement signal catch the REAL dreamer's failures?

The S4 de-risk got 0.78 AUROC separating *synthetic* localized corruptions from
clean patches (relative clean->corrupt cycle-error delta). Here we replay the
SAME machinery on the *generated* dreams from the fine-tuned LIBERO dreamer:

For each held-out item we rebuild the S4-style clip `[obs ... frame_g]` from the
real demo, then form two clips whose ONLY difference is the last frame:
  * realgoal-clip : last frame = real frame_g
  * dream-clip    : last frame = the generated dream of frame_g
We compute per-patch cycle error + (1-visibility) on both, and score three signals
against a per-patch "wrong" label (patches where |dream - real_goal| is high):
  * abs_cyc     : absolute cycle error on the dream clip      (premise: expected BLIND)
  * cyc_delta   : cyc(dream) - cyc(realgoal)  [needs a reference; offline only]
  * invis       : 1 - visibility on the dream clip

Read-out: if abs signals are blind but cyc_delta ranks the wrong patches, the
backbone's correspondence field DOES carry the dreamer's localized error -> a
detector is possible (but needs a reference => motivates a reference-free
cross-consistency signal). If cyc_delta is also flat, the failures are not
geometrically localized.

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    export HF_HOME=/workspace/huggingface_cache/
    CUDA_VISIBLE_DEVICES=0 python experiments/M2_libero_dreamer/s4_on_dreams.py \
        --config <backbone cfg> --ckpt <backbone ckpt> \
        --dreams experiments/M2_libero_dreamer/dreams/dreams.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_SRC = Path("/workspace/code/GD_4D/src")
sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(_REPO_SRC / "eval"))

import s0_backbone_sanity as s0  # noqa: E402
from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402
from data.corruptions import GRID  # noqa: E402

sys.path.insert(0, str(_REPO_SRC / "eval"))
from s4_disagreement_derisk import auroc, cyc_vis_per_patch  # noqa: E402


def build_clip_from_demo(task, demo, g, K=16, size=256):
    hp = LIBERO_ROOT / "libero_spatial" / f"{task}.hdf5"
    traj = LiberoTrajectory(hp, int(demo))
    g = int(min(g, len(traj) - 1))
    idx = np.linspace(0, g, K).round().astype(int)
    idx[-1] = g
    idx = np.unique(idx)
    frames = [cv2.resize(np.ascontiguousarray(traj.frame(i)), (size, size),
                         interpolation=cv2.INTER_LINEAR) for i in idx]
    return np.stack(frames)  # [K,size,size,3] uint8, last = real frame_g


def wrong_label(dream_u8, goal_u8, grid=GRID):
    """Per-patch 'dream is wrong here' label from |dream-goal| excess (matches characterize.py)."""
    from scipy.ndimage import gaussian_filter
    E = gaussian_filter(np.abs(dream_u8.astype(np.float32).mean(2) - goal_u8.astype(np.float32).mean(2)), 2.0)
    H, W = E.shape
    ph, pw = H // grid, W // grid
    Ep = E[: ph * grid, : pw * grid].reshape(grid, ph, grid, pw).mean(axis=(1, 3))
    floor = np.median(Ep)
    mad = np.median(np.abs(Ep - floor)) + 1e-6
    return (Ep > floor + 2 * mad), Ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dreams", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "viz"))
    args = ap.parse_args()

    device = s0._resolve_device(args.device)
    model = s0.load_backbone(args.config, args.ckpt, device)

    d = np.load(args.dreams, allow_pickle=True)
    obs, goal, dream = d["obs"], d["goal"], d["dream"]
    task, demo, g = d["task"], d["demo"], d["g"]
    N = len(obs)
    size = obs.shape[1]

    acc = {"abs_cyc": [], "cyc_delta": [], "invis": [], "invis_delta": [], "label": []}
    per_item = []
    examples = []
    for i in range(N):
        clip = build_clip_from_demo(str(task[i]), int(demo[i]), int(g[i]), size=size)
        real_clip = clip.copy()                       # last frame = real goal
        dream_clip = clip.copy()
        dream_clip[-1] = cv2.resize(dream[i], (size, size), interpolation=cv2.INTER_LINEAR)

        cyc_r, invis_r = cyc_vis_per_patch(model, real_clip)
        cyc_d, invis_d = cyc_vis_per_patch(model, dream_clip)
        lab, Ep = wrong_label(dream[i], goal[i])

        if lab.sum() == 0 or lab.sum() == lab.size:
            per_item.append({"i": i, "task": str(task[i]), "skipped": True})
            continue
        cyc_delta = cyc_d - cyc_r
        invis_delta = invis_d - invis_r
        acc["abs_cyc"].append(cyc_d.ravel())
        acc["cyc_delta"].append(cyc_delta.ravel())
        acc["invis"].append(invis_d.ravel())
        acc["invis_delta"].append(invis_delta.ravel())
        acc["label"].append(lab.ravel())

        # spatial correlation between geometric delta and pixel-error excess
        corr = float(np.corrcoef(cyc_delta.ravel(), Ep.ravel())[0, 1])
        per_item.append({"i": i, "task": str(task[i]),
                         "auroc_delta": auroc(cyc_delta.ravel(), lab.ravel()),
                         "auroc_abs": auroc(cyc_d.ravel(), lab.ravel()),
                         "corr_delta_pixelerr": corr,
                         "n_wrong_patches": int(lab.sum())})
        if len(examples) < 4:
            examples.append((dream[i], goal[i], np.clip(cyc_delta, 0, None), invis_d, lab, str(task[i])))
        print(f"  [{i:2d}] {str(task[i])[:34]:34s} wrong={int(lab.sum()):3d} "
              f"AUROC delta={per_item[-1]['auroc_delta']:.2f} abs={per_item[-1]['auroc_abs']:.2f} "
              f"corr={corr:+.2f}", flush=True)

    for k in acc:
        acc[k] = np.concatenate(acc[k]) if acc[k] else np.array([])
    report = {
        "n_items_scored": int(sum(1 for p in per_item if not p.get("skipped"))),
        "pooled_auroc_abs_cyc": auroc(acc["abs_cyc"], acc["label"]) if acc["label"].size else None,
        "pooled_auroc_cyc_delta": auroc(acc["cyc_delta"], acc["label"]) if acc["label"].size else None,
        "pooled_auroc_invis": auroc(acc["invis"], acc["label"]) if acc["label"].size else None,
        "pooled_auroc_invis_delta": auroc(acc["invis_delta"], acc["label"]) if acc["label"].size else None,
        "mean_corr_delta_pixelerr": float(np.mean([p["corr_delta_pixelerr"] for p in per_item
                                                   if "corr_delta_pixelerr" in p])) if any("corr_delta_pixelerr" in p for p in per_item) else None,
        "per_item": per_item,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "s4_on_dreams_report.json").write_text(json.dumps(report, indent=2))
    _save_examples(examples, out / "s4_on_dreams_examples.png")

    print("\n[report]")
    print(f"  pooled AUROC  abs cyc  : {report['pooled_auroc_abs_cyc']}")
    print(f"  pooled AUROC  cyc_delta: {report['pooled_auroc_cyc_delta']}  (needs a reference)")
    print(f"  pooled AUROC  invis    : {report['pooled_auroc_invis']}")
    print(f"  mean corr(delta, pixel-error): {report['mean_corr_delta_pixelerr']}")
    print(f"[save] {out/'s4_on_dreams_report.json'} + s4_on_dreams_examples.png")


def _save_examples(examples, path):
    if not examples:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(examples)
    fig, ax = plt.subplots(n, 4, figsize=(4 * 2.4, n * 2.4), dpi=120)
    ax = np.atleast_2d(ax)
    titles = ["dream", "real goal", "cycle-error RISE vs real (px)", "1 - visibility"]
    for r, (dream, goal, cyc_delta, invis, lab, task) in enumerate(examples):
        ax[r, 0].imshow(dream); ax[r, 1].imshow(goal)
        im2 = ax[r, 2].imshow(cyc_delta, cmap="magma"); fig.colorbar(im2, ax=ax[r, 2], fraction=0.046)
        im3 = ax[r, 3].imshow(invis, cmap="viridis"); fig.colorbar(im3, ax=ax[r, 3], fraction=0.046)
        for c in (2, 3):
            ax[r, c].contour(lab.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
        ax[r, 0].set_ylabel(task[:22], fontsize=6)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=9)
    fig.suptitle("S4 signal on REAL generated dreams (cyan = |dream-goal| wrong-patch label)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

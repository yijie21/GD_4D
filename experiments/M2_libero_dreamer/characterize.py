"""Characterize the LIBERO dreamer's failure modes: localized vs global.

Question (direction #1): are a *real* dreamer's errors localized (a wrong bowl
pose that the S4 relative-delta head can catch) or global (whole-scene drift that
no localized signal catches)?

For each held-out (obs, real_goal, dream) triple we compute, in pixel space:
  * E   = |dream - real_goal|      -- where the dream is WRONG
  * C_r = |real_goal - obs|        -- what the task SHOULD change (bowl + arm)
  * C_d = |dream - obs|            -- what the dreamer actually changed
and derived localization stats:
  * floor            : median(E) -- the generative texture floor (global offset)
  * excess_conc      : fraction of *above-floor* error mass in the hottest 10% of
                       patches (high => localized; ~0.10 => uniform/global)
  * excess_area_frac : fraction of patches whose error exceeds floor + 2*MAD
  * change_iou       : IoU(C_r>thr, C_d>thr) -- did the dream edit the RIGHT region?

Outputs a colored gallery (obs | real_goal | dream | error-heatmap) + a summary.

Reproduce (env gd4d5090):
    cd /workspace/code/GD_4D
    python experiments/M2_libero_dreamer/characterize.py \
        --dreams experiments/M2_libero_dreamer/dreams/dreams.npz \
        --out experiments/M2_libero_dreamer/viz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    def gaussian_filter(x, sigma):
        return x


def _gray(x):  # [H,W,3] uint8 -> [H,W] float
    return x.astype(np.float32).mean(axis=2)


def _patchify(m, grid=16):
    """[H,W] -> [grid,grid] mean over patches."""
    H, W = m.shape
    ph, pw = H // grid, W // grid
    return m[: ph * grid, : pw * grid].reshape(grid, ph, grid, pw).mean(axis=(1, 3))


def analyze(obs, goal, dream, grid=16):
    E = gaussian_filter(np.abs(_gray(dream) - _gray(goal)), sigma=2.0)
    Cr = gaussian_filter(np.abs(_gray(goal) - _gray(obs)), sigma=2.0)
    Cd = gaussian_filter(np.abs(_gray(dream) - _gray(obs)), sigma=2.0)

    Ep = _patchify(E, grid).ravel()
    floor = float(np.median(Ep))
    mad = float(np.median(np.abs(Ep - floor)) + 1e-6)
    excess = np.clip(Ep - floor, 0, None)
    total = float(excess.sum() + 1e-6)
    k = max(1, int(0.10 * excess.size))
    top = np.sort(excess)[-k:].sum()
    excess_conc = float(top / total)                       # 0.10=uniform .. 1=one patch
    excess_area = float((Ep > floor + 2 * mad).mean())     # frac of patches "hot"

    # change-region overlap (did the dream edit the right area?)
    def binz(m):
        p = _patchify(m, grid)
        return p > (p.mean() + p.std())
    br, bd = binz(Cr), binz(Cd)
    inter = float((br & bd).sum()); union = float((br | bd).sum() + 1e-6)
    change_iou = inter / union

    return {"floor": floor, "excess_conc": excess_conc,
            "excess_area_frac": excess_area, "change_iou": change_iou,
            "mean_abs_err": float(_gray(np.abs(dream.astype(int) - goal.astype(int)).astype(np.uint8)).mean())}, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreams", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--grid", type=int, default=16)
    args = ap.parse_args()

    d = np.load(args.dreams, allow_pickle=True)
    obs, goal, dream = d["obs"], d["goal"], d["dream"]
    tasks = d["task"] if "task" in d else np.array([f"item{i}" for i in range(len(obs))])
    out = Path(args.out) if args.out else Path(args.dreams).parent.parent / "viz"
    out.mkdir(parents=True, exist_ok=True)
    N = len(obs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats, emaps = [], []
    for i in range(N):
        s, E = analyze(obs[i], goal[i], dream[i], grid=args.grid)
        stats.append(s); emaps.append(E)

    # gallery: N rows x 4 cols (obs | real goal | dream | error heatmap)
    ncol = 4
    fig, ax = plt.subplots(N, ncol, figsize=(ncol * 2.4, N * 2.4))
    if N == 1:
        ax = ax[None, :]
    col_titles = ["obs $o_t$", "real goal", "dream", "|dream-goal|"]
    for i in range(N):
        ax[i, 0].imshow(obs[i]); ax[i, 1].imshow(goal[i]); ax[i, 2].imshow(dream[i])
        hm = ax[i, 3].imshow(emaps[i], cmap="inferno", vmin=0, vmax=max(20, emaps[i].max()))
        ax[i, 0].set_ylabel(f"conc={stats[i]['excess_conc']:.2f}\nIoU={stats[i]['change_iou']:.2f}",
                            fontsize=7, rotation=0, ha="right", va="center")
        for c in range(ncol):
            ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
            if i == 0:
                ax[i, c].set_title(col_titles[c], fontsize=10)
    plt.tight_layout()
    fig.savefig(out / "dream_failure_gallery.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # aggregate
    def agg(key):
        v = np.array([s[key] for s in stats])
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max())}
    summary = {k: agg(k) for k in ["excess_conc", "excess_area_frac", "change_iou",
                                   "floor", "mean_abs_err"]}
    summary["n"] = N
    (out / "characterize_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[chars] N={N}")
    print(f"  excess_conc      (localized>0.3, uniform~0.1): mean {summary['excess_conc']['mean']:.3f} "
          f"[{summary['excess_conc']['min']:.2f},{summary['excess_conc']['max']:.2f}]")
    print(f"  excess_area_frac (hot-patch fraction)        : mean {summary['excess_area_frac']['mean']:.3f}")
    print(f"  change_iou       (dream edits right region)  : mean {summary['change_iou']['mean']:.3f}")
    print(f"  floor (px, generative texture offset)        : mean {summary['floor']['mean']:.2f}")
    conc = summary["excess_conc"]["mean"]
    verdict = ("LOCALIZED (excess error concentrated -> S4 relative-delta applies)"
               if conc > 0.30 else
               "MIXED/GLOBAL (excess error spread -> localized signal may miss)")
    print(f"  VERDICT: {verdict}")
    print(f"[save] {out/'dream_failure_gallery.png'} + characterize_summary.json")


if __name__ == "__main__":
    main()

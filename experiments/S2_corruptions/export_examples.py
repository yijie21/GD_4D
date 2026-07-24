"""Export S2 corruption example figures: for each generator, show
`original goal → corruption (+ pixel region) → 16×16 patch label`, on two LIBERO scenes.

Run:
  /workspace/miniconda3/envs/gd4d5090/bin/python experiments/S2_corruptions/export_examples.py
Output: experiments/S2_corruptions/viz/examples_scene{1,2}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402
from data.corruptions import (  # noqa: E402
    GRID, cross_episode_composite, cut_paste, mismatched_frame, tps_warp,
)
from data.sam2_masks import Sam2MaskGenerator  # noqa: E402

OUT = Path(__file__).resolve().parent / "viz"


def goal256(traj) -> np.ndarray:
    return cv2.resize(np.ascontiguousarray(traj.frame(len(traj) - 1)), (256, 256), interpolation=cv2.INTER_LINEAR)


def region_overlay(sample) -> np.ndarray:
    img = sample.frame.copy()
    reg = sample.region_mask.astype(np.uint8)
    edge = (reg - cv2.erode(reg, np.ones((3, 3), np.uint8))).astype(bool)
    img[reg.astype(bool)] = (0.75 * img[reg.astype(bool)] + 0.25 * np.array([255, 40, 40])).astype(np.uint8)
    img[edge] = [255, 40, 40]
    return img


def patch_overlay(sample) -> np.ndarray:
    img = sample.frame.copy()
    H, W = img.shape[:2]
    up = cv2.resize(sample.patch_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[up] = (0.55 * img[up] + 0.45 * np.array([255, 235, 0])).astype(np.uint8)
    for i in range(1, GRID):
        y, x = round(i * H / GRID), round(i * W / GRID)
        img[y, :] = (0.6 * img[y, :] + 0.4 * np.array([90, 90, 90])).astype(np.uint8)
        img[:, x] = (0.6 * img[:, x] + 0.4 * np.array([90, 90, 90])).astype(np.uint8)
    return img


def scene_figure(target, donor, tmasks, dmasks, seed, out_path):
    rng = np.random.default_rng(seed)
    m = lambda i: tmasks[i % len(tmasks)]                      # cycle objects for variety
    rows = [
        ("cut-paste · remove",    cut_paste(target, m(0), "remove")),
        ("cut-paste · relocate",  cut_paste(target, m(1), "relocate", rng)),
        ("cut-paste · duplicate", cut_paste(target, m(2), "duplicate", rng)),
        ("tps warp",              tps_warp(target, m(3), rng=rng)),
        ("mismatched frame",      mismatched_frame(target, donor)),
        ("cross-episode composite", cross_episode_composite(target, donor, dmasks[0], rng)),
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.2, 2.7 * len(rows)), dpi=120)
    cols = ["original goal", "corruption + region", "16×16 patch label"]
    imgs0 = [target, region_overlay(rows[0][1]), patch_overlay(rows[0][1])]  # for column titles only
    for r, (name, s) in enumerate(rows):
        panels = [target, region_overlay(s), patch_overlay(s)]
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(panels[c]); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r, 0].set_ylabel(f"{name}\n{int(s.patch_mask.sum())}/{GRID*GRID}", fontsize=10, rotation=90, labelpad=8)
    fig.suptitle("S2 corruption generators on a LIBERO goal frame", fontsize=13, y=0.997)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {name: int(s.patch_mask.sum()) for name, s in rows}


def main() -> int:
    files = sorted((LIBERO_ROOT / "libero_spatial").glob("*.hdf5"))
    if len(files) < 4:
        raise SystemExit("need ≥4 LIBERO files for two scenes")
    mg = Sam2MaskGenerator()
    for scene, (ti, di) in enumerate([(0, 1), (2, 3)], start=1):
        target = goal256(LiberoTrajectory(files[ti], 0))
        donor = goal256(LiberoTrajectory(files[di], 0))
        tmasks, dmasks = mg.object_masks(target), mg.object_masks(donor)
        out = OUT / f"examples_scene{scene}.png"
        counts = scene_figure(target, donor, tmasks, dmasks, seed=scene, out_path=out)
        print(f"scene{scene}: target_masks={len(tmasks)} donor_masks={len(dmasks)} | {counts} | wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S2 corruptions demo — run all four generators on real LIBERO goal frames with SAM 2 masks,
and render a montage showing the corrupted frame + pixel region (red) + 16×16 patch label (yellow).

Run:
  /workspace/miniconda3/envs/gd4d5090/bin/python experiments/S2_corruptions/demo.py
Output: experiments/S2_corruptions/viz/corruptions_demo.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo/src on path

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
    g = traj.frame(len(traj) - 1)                      # last frame = reached goal
    return cv2.resize(np.ascontiguousarray(g), (256, 256), interpolation=cv2.INTER_LINEAR)


def overlay(frame: np.ndarray, sample) -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]
    if sample is not None:
        up = cv2.resize(sample.patch_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        img[up] = (0.6 * img[up] + 0.4 * np.array([255, 235, 0])).astype(np.uint8)
        reg = sample.region_mask.astype(np.uint8)
        edge = (reg - cv2.erode(reg, np.ones((3, 3), np.uint8))).astype(bool)
        img[edge] = [255, 40, 40]
    for i in range(1, GRID):
        y, x = round(i * H / GRID), round(i * W / GRID)
        img[y, :] = (0.6 * img[y, :] + 0.4 * np.array([90, 90, 90])).astype(np.uint8)
        img[:, x] = (0.6 * img[:, x] + 0.4 * np.array([90, 90, 90])).astype(np.uint8)
    return img


def main() -> int:
    files = sorted((LIBERO_ROOT / "libero_spatial").glob("*.hdf5"))
    if len(files) < 2:
        raise SystemExit(f"need ≥2 LIBERO files under {LIBERO_ROOT/'libero_spatial'}")
    target = goal256(LiberoTrajectory(files[0], 0))
    donor = goal256(LiberoTrajectory(files[1], 0))

    mg = Sam2MaskGenerator()
    tmasks = mg.object_masks(target)
    dmasks = mg.object_masks(donor)
    print(f"SAM2 masks: target={len(tmasks)}  donor={len(dmasks)}")
    if not tmasks or not dmasks:
        raise SystemExit("SAM 2 returned no object masks")

    rng = np.random.default_rng(0)
    second = tmasks[1] if len(tmasks) > 1 else tmasks[0]
    samples = [
        cut_paste(target, tmasks[0], "remove"),
        cut_paste(target, tmasks[0], "relocate", rng),
        cut_paste(target, second, "duplicate", rng),
        tps_warp(target, tmasks[0], rng=rng),
        mismatched_frame(target, donor),
        cross_episode_composite(target, donor, dmasks[0], rng),
    ]

    panels = [("original goal", target, None)] + [(s.ctype, s.frame, s) for s in samples]
    cols, rows = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.3 * rows), dpi=120)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, frame, s) in zip(axes.flat, panels):
        ax.imshow(overlay(frame, s))
        n = "" if s is None else f"   ·   {int(s.patch_mask.sum())}/{GRID*GRID} patches"
        ax.set_title(f"{title}{n}", fontsize=10)
    fig.suptitle("S2 corruptions on a LIBERO goal frame   (red = region, yellow = 16×16 label)", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "corruptions_demo.png", bbox_inches="tight")
    plt.close(fig)

    print("patch counts:", {s.ctype: int(s.patch_mask.sum()) for s in samples})
    print("wrote", OUT / "corruptions_demo.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

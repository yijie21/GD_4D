"""S2 corruption generators — make labeled "bad dreams" from a veridical goal frame.

Cheap-negative design (plan §M0/S2, **no diffusion editor**): each generator takes a frame and an
object mask (from SAM 2, or any source) and returns a :class:`CorruptionSample` with the corrupted
frame, the pixel-level corrupted region, and the **16×16 patch label** (a patch is corrupted if
**≥25%** of its area overlaps the region). Optimized for geometric **feature coverage**
(displacement / occlusion / scale), not photorealism (§5.2).

The four generators:
  - `cut_paste`  — remove / relocate / duplicate an object (SAM 2 mask + copy-paste + inpaint fill)
  - `tps_warp`   — TPS-style elastic warp localized to an object mask
  - `mismatched_frame` — inject a wrong reached-frame (whole-frame negative)
  - `cross_episode_composite` — paste a foreign object onto the frame

Masks are `[H,W]` bool; frames are `[H,W,3]` uint8 RGB. Generators are pure given an `rng`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GRID = 16
PATCH_THRESH = 0.25


@dataclass
class CorruptionSample:
    frame: np.ndarray        # [H, W, 3] uint8 — corrupted goal frame
    region_mask: np.ndarray  # [H, W] bool     — pixel-level corrupted region
    patch_mask: np.ndarray   # [GRID, GRID] bool — patch label (True = corrupted)
    ctype: str               # e.g. "cutpaste_relocate", "tps", "mismatch", "composite"
    provenance: str = "synthetic"


def region_to_patch_labels(region_mask: np.ndarray, grid: int = GRID, thresh: float = PATCH_THRESH) -> np.ndarray:
    """[H,W] bool region → [grid,grid] bool: patch True iff (overlap area / patch area) ≥ thresh."""
    H, W = region_mask.shape
    r = region_mask.astype(np.float32)
    ys = np.linspace(0, H, grid + 1).round().astype(int)
    xs = np.linspace(0, W, grid + 1).round().astype(int)
    out = np.zeros((grid, grid), bool)
    for i in range(grid):
        for j in range(grid):
            cell = r[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if cell.size and cell.mean() >= thresh:
                out[i, j] = True
    return out


def _sample(frame: np.ndarray, region: np.ndarray, ctype: str, prov: str = "synthetic") -> CorruptionSample:
    region = region.astype(bool)
    return CorruptionSample(frame=frame, region_mask=region,
                            patch_mask=region_to_patch_labels(region), ctype=ctype, provenance=prov)


def _paste(canvas: np.ndarray, src_frame: np.ndarray, src_mask: np.ndarray, dy: int, dx: int):
    """Copy the masked object from src_frame into canvas, shifted by (dy,dx). No wraparound."""
    H, W = src_mask.shape
    ys, xs = np.where(src_mask)
    ty, tx = ys + dy, xs + dx
    ok = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
    canvas[ty[ok], tx[ok]] = src_frame[ys[ok], xs[ok]]
    pm = np.zeros((H, W), bool)
    pm[ty[ok], tx[ok]] = True
    return canvas, pm


def _inpaint(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(np.ascontiguousarray(frame), (mask.astype(np.uint8) * 255), 3, cv2.INPAINT_TELEA)


def _rand_shift(mask: np.ndarray, rng: np.random.Generator, frac=(0.15, 0.40)):
    H, W = mask.shape
    mag = rng.uniform(*frac)
    ang = rng.uniform(0, 2 * np.pi)
    return int(round(mag * H * np.sin(ang))), int(round(mag * W * np.cos(ang)))


def cut_paste(frame: np.ndarray, obj_mask: np.ndarray, mode: str = "relocate",
              rng: np.random.Generator | None = None) -> CorruptionSample:
    """Remove / relocate / duplicate the masked object. Region = affected pixels (exact)."""
    if mode not in {"remove", "relocate", "duplicate"}:
        raise ValueError(f"mode must be remove|relocate|duplicate, got {mode!r}")
    rng = rng or np.random.default_rng()
    obj_mask = obj_mask.astype(bool)
    H, W = obj_mask.shape
    if not obj_mask.any():
        return _sample(frame.copy(), np.zeros((H, W), bool), f"cutpaste_{mode}")

    if mode == "remove":
        out, region = _inpaint(frame, obj_mask), obj_mask
    elif mode == "relocate":
        dy, dx = _rand_shift(obj_mask, rng)
        out, pm = _paste(_inpaint(frame, obj_mask).copy(), frame, obj_mask, dy, dx)
        region = obj_mask | pm
    else:  # duplicate — keep original, add a shifted copy
        dy, dx = _rand_shift(obj_mask, rng)
        out, pm = _paste(frame.copy(), frame, obj_mask, dy, dx)
        region = pm
    return _sample(out, region, f"cutpaste_{mode}")


def tps_warp(frame: np.ndarray, obj_mask: np.ndarray, disp_px=(5.0, 20.0), n_ctrl: int = 4,
             rng: np.random.Generator | None = None) -> CorruptionSample:
    """TPS-style elastic warp: smooth control-point displacement field localized to the object.

    Control-point magnitudes ~U[disp_px] (px); the field is cubic-upsampled and masked to a dilated
    object region so only the object bends. Region = the dilated mask (where content changed).
    """
    rng = rng or np.random.default_rng()
    H, W = obj_mask.shape
    interp = cv2.INTER_CUBIC if n_ctrl >= 4 else cv2.INTER_LINEAR
    mag = rng.uniform(disp_px[0], disp_px[1], size=(n_ctrl, n_ctrl)).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, size=(n_ctrl, n_ctrl)).astype(np.float32)
    dx = cv2.resize(mag * np.cos(ang), (W, H), interpolation=interp)
    dy = cv2.resize(mag * np.sin(ang), (W, H), interpolation=interp)

    k = max(3, int(round(disp_px[1])) | 1)                        # odd kernel ~ max displacement
    dil = cv2.dilate(obj_mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    soft = cv2.GaussianBlur(dil.astype(np.float32), (0, 0), sigmaX=k / 2.0)
    dx *= soft
    dy *= soft

    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    warped = cv2.remap(np.ascontiguousarray(frame), xx + dx, yy + dy,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = frame.copy()
    out[dil] = warped[dil]
    return _sample(out, dil, "tps")


def mismatched_frame(target_frame: np.ndarray, other_frame: np.ndarray) -> CorruptionSample:
    """Inject a wrong reached-frame (from another episode/time/task). Whole frame is corrupted."""
    H, W = target_frame.shape[:2]
    if other_frame.shape[:2] != (H, W):
        other_frame = cv2.resize(other_frame, (W, H), interpolation=cv2.INTER_LINEAR)
    return _sample(other_frame.copy(), np.ones((H, W), bool), "mismatch")


def cross_episode_composite(target_frame: np.ndarray, donor_frame: np.ndarray, donor_mask: np.ndarray,
                            rng: np.random.Generator | None = None) -> CorruptionSample:
    """Paste a foreign object (donor_mask on donor_frame) onto target_frame. Region = pasted mask."""
    rng = rng or np.random.default_rng()
    donor_mask = donor_mask.astype(bool)
    H, W = target_frame.shape[:2]
    if donor_frame.shape[:2] != (H, W):
        donor_frame = cv2.resize(donor_frame, (W, H), interpolation=cv2.INTER_LINEAR)
        donor_mask = cv2.resize(donor_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.where(donor_mask)
    if ys.size == 0:
        return _sample(target_frame.copy(), np.zeros((H, W), bool), "composite")
    cy, cx = int(ys.mean()), int(xs.mean())
    ty = int(rng.integers(int(0.30 * H), int(0.70 * H)))
    tx = int(rng.integers(int(0.30 * W), int(0.70 * W)))
    out, pm = _paste(target_frame.copy(), donor_frame, donor_mask, ty - cy, tx - cx)
    return _sample(out, pm, "composite")

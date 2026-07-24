"""S2 corruption-generator tests — pure geometry, no SAM 2 / GPU (always-on).

SAM 2 end-to-end on real frames is exercised by experiments/S2_corruptions/demo.py, not here,
so `pytest` stays fast and dependency-free.
"""
from __future__ import annotations

import numpy as np

from data.corruptions import (
    GRID,
    cross_episode_composite,
    cut_paste,
    mismatched_frame,
    region_to_patch_labels,
    tps_warp,
)


def _frame(H=64, W=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), np.uint8)


def _square_mask(H=64, W=64, y0=12, y1=28, x0=12, x1=28):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_patch_labels_shape_and_full_patch():
    m = np.zeros((64, 64), bool)
    m[0:4, 0:4] = True                         # exactly one 4×4 patch (grid 16 on 64px)
    lab = region_to_patch_labels(m)
    assert lab.shape == (GRID, GRID)
    assert lab[0, 0] and lab.sum() == 1


def test_patch_threshold_boundary():
    m = np.zeros((64, 64), bool)
    m[0:1, 0:2] = True                          # 2/16 px = 12.5% < 25% → not labeled
    assert region_to_patch_labels(m).sum() == 0
    m[:] = False
    m[0:2, 0:2] = True                          # 4/16 px = 25% ≥ 25% → labeled
    assert region_to_patch_labels(m)[0, 0]


def test_cut_paste_all_modes():
    f, m = _frame(), _square_mask()
    for mode in ("remove", "relocate", "duplicate"):
        s = cut_paste(f, m, mode=mode, rng=np.random.default_rng(1))
        assert s.frame.shape == f.shape and s.frame.dtype == np.uint8
        assert s.region_mask.shape == (64, 64) and s.region_mask.any()
        assert s.patch_mask.shape == (GRID, GRID)
        assert np.array_equal(s.patch_mask, region_to_patch_labels(s.region_mask))
        assert s.ctype == f"cutpaste_{mode}"


def test_cut_paste_remove_region_is_object():
    f, m = _frame(), _square_mask()
    s = cut_paste(f, m, mode="remove")
    assert np.array_equal(s.region_mask, m)     # remove touches exactly the object
    assert not np.array_equal(s.frame, f)       # pixels changed (inpainted)


def test_cut_paste_empty_mask_is_noop():
    f = _frame()
    s = cut_paste(f, np.zeros((64, 64), bool), mode="relocate")
    assert np.array_equal(s.frame, f) and s.patch_mask.sum() == 0


def test_tps_warp_localized_and_deterministic():
    f, m = _frame(), _square_mask()
    s1 = tps_warp(f, m, rng=np.random.default_rng(2))
    s2 = tps_warp(f, m, rng=np.random.default_rng(2))
    assert s1.frame.shape == f.shape and s1.ctype == "tps"
    assert s1.region_mask.any()
    assert np.array_equal(s1.frame, s2.frame)   # seeded → deterministic
    # warp is localized: pixels far from the object are untouched
    far = np.zeros((64, 64), bool); far[55:64, 55:64] = True
    assert np.array_equal(s1.frame[far], f[far])


def test_mismatched_frame():
    f, o = _frame(seed=0), _frame(seed=9)
    s = mismatched_frame(f, o)
    assert np.array_equal(s.frame, o)
    assert s.patch_mask.all() and s.ctype == "mismatch"


def test_cross_episode_composite():
    tgt, donor, dm = _frame(seed=0), _frame(seed=5), _square_mask()
    s = cross_episode_composite(tgt, donor, dm, rng=np.random.default_rng(3))
    assert s.frame.shape == tgt.shape and s.ctype == "composite"
    assert s.region_mask.any()
    assert np.array_equal(s.patch_mask, region_to_patch_labels(s.region_mask))


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

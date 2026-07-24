"""Unit tests for S1 sliding-window logic (plan §M0/S1)."""
from __future__ import annotations

import numpy as np
import pytest

from data.dataset import TupleDataset
from data.schema import WindowConfig
from data.trajectory import SyntheticTrajectory
from data.windowing import build_tuple, build_tuples, plan_tuples

CFG = WindowConfig(h=8, n=20, stride=4)


# ----------------------------- plan_tuples (pure) -----------------------------

def test_single_tuple_at_minimum_length():
    # L = h + n = 28 -> exactly one tuple at t = h-1 = 7, goal = 27
    specs = plan_tuples("ep", length=CFG.min_length, cfg=CFG)
    assert len(specs) == 1
    s = specs[0]
    assert s.t == 7
    assert s.obs_indices == tuple(range(0, 8))        # [0..7]
    assert s.waypoint_indices == tuple(range(8, 28))  # [8..27]
    assert s.goal_index == 27
    assert s.h == 8 and s.n == 20


def test_too_short_yields_nothing():
    specs = plan_tuples("ep", length=CFG.min_length - 1, cfg=CFG)  # L = 27
    assert specs == []


def test_stride_and_count():
    # L = 32 -> t in range(7, 32-20+1=13, 4) = [7, 11] -> 2 tuples
    specs = plan_tuples("ep", length=32, cfg=CFG)
    assert [s.t for s in specs] == [7, 11]
    assert specs[-1].goal_index == 31  # last goal must be within bounds (<= L-1 = 31)


def test_shapes_and_bounds_are_consistent():
    specs = plan_tuples("ep", length=100, cfg=CFG)
    assert len(specs) > 0
    for s in specs:
        assert len(s.obs_indices) == CFG.h
        assert len(s.waypoint_indices) == CFG.n
        assert s.obs_indices[-1] == s.t                     # history ends at t
        assert s.waypoint_indices[0] == s.t + 1             # waypoints start at t+1
        assert s.waypoint_indices[-1] == s.goal_index       # goal is the last waypoint
        assert s.goal_index == s.t + CFG.n
        assert min(s.obs_indices) >= 0                      # no negative indices
        assert max(s.waypoint_indices) <= 99               # within [0, L)


@pytest.mark.parametrize("h,n,stride", [(1, 1, 1), (4, 10, 2), (8, 20, 4), (16, 5, 3)])
def test_configs_never_go_out_of_bounds(h, n, stride):
    cfg = WindowConfig(h=h, n=n, stride=stride)
    L = 64
    for s in plan_tuples("ep", length=L, cfg=cfg):
        assert s.obs_indices[0] >= 0
        assert s.goal_index <= L - 1


# ----------------------------- materialization -----------------------------

def test_build_tuple_frame_shapes_and_goal_alias():
    traj = SyntheticTrajectory(length=40, hw=(16, 16))
    spec = plan_tuples(traj.episode_id, len(traj), CFG)[0]
    tup = build_tuple(traj, spec)
    assert tup.obs_frames.shape == (CFG.h, 16, 16, 3)
    assert tup.waypoint_frames.shape == (CFG.n, 16, 16, 3)
    assert tup.goal_frame.shape == (16, 16, 3)
    assert tup.obs_frames.dtype == np.uint8
    # goal frame == last waypoint frame == the actual reached frame
    np.testing.assert_array_equal(tup.goal_frame, tup.waypoint_frames[-1])
    np.testing.assert_array_equal(tup.goal_frame, traj.frame(spec.goal_index))
    assert tup.intrinsics.shape == (3, 3) and tup.extrinsics.shape == (4, 4)
    assert tup.instruction == traj.instruction()


def test_build_tuples_matches_plan_count():
    traj = SyntheticTrajectory(length=60)
    n_planned = len(plan_tuples(traj.episode_id, len(traj), CFG))
    n_built = sum(1 for _ in build_tuples(traj, CFG))
    assert n_built == n_planned > 0


# ----------------------------- dataset -----------------------------

def test_dataset_flattens_across_trajectories():
    trajs = [SyntheticTrajectory(episode_id=f"ep{i}", length=40 + 4 * i, seed=i)
             for i in range(3)]
    ds = TupleDataset(trajs, CFG)
    expected = sum(len(plan_tuples(t.episode_id, len(t), CFG)) for t in trajs)
    assert len(ds) == expected
    tup = ds[0]
    assert tup.obs_frames.shape[0] == CFG.h
    assert len(ds.specs) == len(ds)


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        WindowConfig(h=0)


if __name__ == "__main__":  # allow: python src/data/tests/test_windowing.py
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

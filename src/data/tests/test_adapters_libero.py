"""Real-data smoke tests for the LIBERO adapter (S1 end-to-end on actual pixels).

These are gated: skipped where h5py or the LIBERO dataset are absent, so the always-on
synthetic suite (test_windowing.py) stays dependency-free. On the box they confirm S1 runs
on real frames — the prerequisite for S2.
"""
from __future__ import annotations

import numpy as np
import pytest

from data.adapters import LIBERO_ROOT, LiberoTrajectory, iter_libero_episodes
from data.schema import WindowConfig
from data.windowing import build_tuples, plan_tuples

pytest.importorskip("h5py")

_SUITE = LIBERO_ROOT / "libero_spatial"
_FILES = sorted(_SUITE.glob("*.hdf5")) if _SUITE.exists() else []
pytestmark = pytest.mark.skipif(not _FILES, reason=f"LIBERO data not found under {_SUITE}")


def test_libero_trajectory_basics():
    traj = LiberoTrajectory(_FILES[0], demo_index=0)
    assert len(traj) > 0
    f0 = traj.frame(0)
    assert f0.ndim == 3 and f0.shape[2] == 3 and f0.dtype == np.uint8
    assert traj.intrinsics().shape == (3, 3)
    assert traj.extrinsics().shape == (4, 4)
    assert isinstance(traj.instruction(), str) and traj.instruction()          # non-empty
    assert traj.episode_id.startswith("libero/")


def test_libero_build_tuples_end_to_end():
    traj = LiberoTrajectory(_FILES[0], demo_index=0)
    cfg = WindowConfig()                                    # h=8, n=20, stride=4
    if len(traj) < cfg.min_length:
        pytest.skip("episode shorter than one window")

    specs = plan_tuples(traj.episode_id, len(traj), cfg)
    assert specs, "expected at least one tuple from a full LIBERO episode"

    tuples = list(build_tuples(traj, cfg))
    assert len(tuples) == len(specs)
    tup = tuples[0]
    H, W = traj.frame(0).shape[:2]
    assert tup.obs_frames.shape == (cfg.h, H, W, 3)
    assert tup.waypoint_frames.shape == (cfg.n, H, W, 3)
    # goal is the last waypoint, and its index is t + n
    assert np.array_equal(tup.goal_frame, tup.waypoint_frames[-1])
    assert tup.spec.goal_index == tup.spec.t + cfg.n
    assert tup.instruction == traj.instruction()


def test_iter_libero_episodes_caps():
    eps = list(iter_libero_episodes(suite="libero_spatial", max_files=2, max_demos_per_file=3))
    assert 0 < len(eps) <= 6
    assert all(isinstance(e, LiberoTrajectory) for e in eps)
    assert all(len(e) > 0 for e in eps)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

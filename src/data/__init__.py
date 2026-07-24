"""GD-4D data pipeline (Milestone M0).

S1 (build training tuples) is implemented here; S2 (corrupt & label) will be added next.
"""
from __future__ import annotations

from .adapters import LiberoTrajectory, iter_libero_episodes
from .corruptions import (
    CorruptionSample,
    cross_episode_composite,
    cut_paste,
    mismatched_frame,
    region_to_patch_labels,
    tps_warp,
)
from .dataset import TupleDataset
from .schema import TrainingTuple, TupleSpec, WindowConfig
from .trajectory import SyntheticTrajectory, Trajectory
from .windowing import build_tuple, build_tuples, plan_tuples

__all__ = [
    "WindowConfig",
    "TupleSpec",
    "TrainingTuple",
    "Trajectory",
    "SyntheticTrajectory",
    "plan_tuples",
    "build_tuple",
    "build_tuples",
    "TupleDataset",
    "LiberoTrajectory",
    "iter_libero_episodes",
    "CorruptionSample",
    "region_to_patch_labels",
    "cut_paste",
    "tps_warp",
    "mismatched_frame",
    "cross_episode_composite",
]

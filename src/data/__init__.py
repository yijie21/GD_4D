"""GD-4D data pipeline (Milestone M0).

S1 (build training tuples) is implemented here; S2 (corrupt & label) will be added next.
"""
from __future__ import annotations

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
]

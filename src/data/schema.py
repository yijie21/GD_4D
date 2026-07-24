"""S1 data structures — the training-tuple schema.

A GD-4D training tuple (plan §M0/S1) captures, for one "current time" ``t`` inside a
trajectory:

    * the observation history ``o_{t-h:t}``  (h frames ending at t, inclusive)
    * the intermediate / waypoint frames ``f_{t+1:t+n}``  (n frames; the last is the goal)
    * the reached goal frame ``f_{t+n}``  (the veridical "perfect dream")
    * the fixed-camera intrinsics/extrinsics and the task instruction

``TupleSpec`` is the *index-level* description (no pixels) — cheap to enumerate and test.
``TrainingTuple`` is the *materialized* tuple (pixels + camera + instruction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window hyperparameters for S1 (defaults from the implementation plan)."""

    h: int = 8       # observation-history length: frames [t-h+1 .. t]
    n: int = 20      # sub-goal horizon: the goal is frame t+n
    stride: int = 4  # step between successive current-times t

    def __post_init__(self) -> None:
        if self.h < 1 or self.n < 1 or self.stride < 1:
            raise ValueError(f"h, n, stride must all be >= 1 (got {self})")

    @property
    def min_length(self) -> int:
        """Shortest trajectory that yields at least one tuple (need t-h+1>=0 and t+n<=L-1)."""
        return self.h + self.n  # earliest t = h-1, latest goal index = (h-1)+n = h+n-1 <= L-1


@dataclass(frozen=True)
class TupleSpec:
    """Index-level description of one training tuple (no pixels)."""

    episode_id: str
    t: int                              # current time index
    obs_indices: tuple[int, ...]        # length h: [t-h+1 .. t]
    waypoint_indices: tuple[int, ...]   # length n: [t+1 .. t+n]  (goal is the last)
    goal_index: int                     # == t + n == waypoint_indices[-1]

    @property
    def h(self) -> int:
        return len(self.obs_indices)

    @property
    def n(self) -> int:
        return len(self.waypoint_indices)


@dataclass
class TrainingTuple:
    """Materialized S1 tuple: pixels + camera + instruction.

    Frame arrays are ``uint8`` in ``[H, W, 3]`` (channels-last, RGB). ``goal_frame`` is
    identical to ``waypoint_frames[-1]`` and kept as a convenience alias.
    """

    spec: TupleSpec
    obs_frames: np.ndarray        # [h, H, W, 3] uint8
    waypoint_frames: np.ndarray   # [n, H, W, 3] uint8  (goal == waypoint_frames[-1])
    goal_frame: np.ndarray        # [H, W, 3] uint8
    intrinsics: np.ndarray        # [3, 3] float  (pinhole K)
    extrinsics: np.ndarray        # [4, 4] float  (fixed third-person camera, world<-cam)
    instruction: str

"""S1 sliding-window logic: enumerate and materialize training tuples.

``plan_tuples`` is a *pure* function over (episode length, config) — it produces the
index-level :class:`TupleSpec` list and is what the unit tests exercise. ``build_tuple`` /
``build_tuples`` then read pixels from a :class:`Trajectory`.

Windowing rule (plan §M0/S1), for current time ``t`` with ``h`` history and horizon ``n``:

    obs      = [t-h+1 .. t]      (h frames, ending at t)
    waypoints= [t+1   .. t+n]    (n frames, the last being the goal)
    goal     = t+n

Validity: ``t-h+1 >= 0`` and ``t+n <= L-1``  =>  ``t in [h-1 .. L-1-n]`` stepped by stride.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .schema import TrainingTuple, TupleSpec, WindowConfig
from .trajectory import Trajectory


def plan_tuples(episode_id: str, length: int, cfg: WindowConfig) -> list[TupleSpec]:
    """Enumerate the index-level tuple specs for one trajectory of ``length`` frames.

    Returns an empty list if the trajectory is shorter than ``cfg.min_length``.
    """
    specs: list[TupleSpec] = []
    t_min = cfg.h - 1
    t_max = length - 1 - cfg.n          # inclusive upper bound on t
    for t in range(t_min, t_max + 1, cfg.stride):
        obs = tuple(range(t - cfg.h + 1, t + 1))          # length h
        waypoints = tuple(range(t + 1, t + cfg.n + 1))    # length n
        specs.append(
            TupleSpec(
                episode_id=episode_id,
                t=t,
                obs_indices=obs,
                waypoint_indices=waypoints,
                goal_index=t + cfg.n,
            )
        )
    return specs


def build_tuple(traj: Trajectory, spec: TupleSpec) -> TrainingTuple:
    """Materialize one :class:`TupleSpec` into a :class:`TrainingTuple` (reads pixels)."""
    obs_frames = np.stack([traj.frame(i) for i in spec.obs_indices])
    waypoint_frames = np.stack([traj.frame(i) for i in spec.waypoint_indices])
    goal_frame = traj.frame(spec.goal_index)
    return TrainingTuple(
        spec=spec,
        obs_frames=obs_frames,
        waypoint_frames=waypoint_frames,
        goal_frame=goal_frame,
        intrinsics=traj.intrinsics(),
        extrinsics=traj.extrinsics(),
        instruction=traj.instruction(),
    )


def build_tuples(traj: Trajectory, cfg: WindowConfig) -> Iterator[TrainingTuple]:
    """Yield every materialized training tuple for one trajectory."""
    for spec in plan_tuples(traj.episode_id, len(traj), cfg):
        yield build_tuple(traj, spec)

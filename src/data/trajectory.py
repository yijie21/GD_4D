"""Trajectory interface + a synthetic trajectory for tests.

A ``Trajectory`` is the minimal, dataset-agnostic view the S1 windowing needs: a sequence
of frames plus a fixed camera and a task instruction. Concrete datasets (CALVIN, LIBERO,
BridgeData V2) implement this via adapters in :mod:`data.adapters`.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Trajectory(Protocol):
    """One recorded episode with a fixed third-person camera.

    Implementations must expose ``episode_id`` and the methods below. Frames are
    ``uint8`` ``[H, W, 3]`` RGB; the camera is assumed static across the episode
    (BridgeData-V2 motion episodes are dropped upstream — see the plan §3).
    """

    episode_id: str

    def __len__(self) -> int: ...
    def frame(self, i: int) -> np.ndarray: ...   # [H, W, 3] uint8
    def intrinsics(self) -> np.ndarray: ...        # [3, 3]
    def extrinsics(self) -> np.ndarray: ...        # [4, 4]
    def instruction(self) -> str: ...


class SyntheticTrajectory:
    """Deterministic synthetic trajectory — lets us test S1 with no dataset present.

    Frames are random-but-reproducible (seeded). Camera is an arbitrary pinhole with an
    identity extrinsic. Used by the unit tests and as a smoke-test fixture.
    """

    def __init__(
        self,
        episode_id: str = "syn-0",
        length: int = 32,
        hw: tuple[int, int] = (32, 32),
        instruction: str = "synthetic pick task",
        seed: int = 0,
    ) -> None:
        if length < 1:
            raise ValueError("length must be >= 1")
        self.episode_id = episode_id
        self._len = length
        self._instr = instruction
        H, W = hw
        rng = np.random.default_rng(seed)
        self._frames = rng.integers(0, 256, size=(length, H, W, 3), dtype=np.uint8)
        # arbitrary but valid pinhole intrinsics + identity extrinsics
        self._K = np.array([[float(H), 0.0, W / 2.0],
                            [0.0, float(H), H / 2.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self._E = np.eye(4, dtype=np.float64)

    def __len__(self) -> int:
        return self._len

    def frame(self, i: int) -> np.ndarray:
        return self._frames[i]

    def intrinsics(self) -> np.ndarray:
        return self._K

    def extrinsics(self) -> np.ndarray:
        return self._E

    def instruction(self) -> str:
        return self._instr

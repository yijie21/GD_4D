"""Dataset-specific adapters that expose each dataset as a :class:`Trajectory`.

These are **stubs** — fill them in when the datasets land under ``data/`` (the symlink to
``/workspace/datasets/gd_4d_data``; see project CLAUDE.md). Each adapter must return, per
episode: RGB frames ``[H, W, 3]`` uint8, fixed-camera intrinsics ``[3,3]`` and extrinsics
``[4,4]``, and the task instruction.

Plan §3 dataset notes:
    * CALVIN (ABCD splits)
    * LIBERO (all 4 task suites)
    * BridgeData V2 — **static-camera episodes only** (drop camera-motion episodes so the
      pinhole model stays exact)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Root of the datasets tree (the repo `data/` symlink -> /workspace/datasets/gd_4d_data).
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


class _NotImplementedTrajectory:
    """Placeholder that documents the required interface until data is wired up."""

    episode_id: str = "unset"

    def __len__(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError(self._msg())

    def frame(self, i: int) -> np.ndarray:  # pragma: no cover - stub
        raise NotImplementedError(self._msg())

    def intrinsics(self) -> np.ndarray:  # pragma: no cover - stub
        raise NotImplementedError(self._msg())

    def extrinsics(self) -> np.ndarray:  # pragma: no cover - stub
        raise NotImplementedError(self._msg())

    def instruction(self) -> str:  # pragma: no cover - stub
        raise NotImplementedError(self._msg())

    @classmethod
    def _msg(cls) -> str:
        return (
            f"{cls.__name__} is a stub. Download the dataset under {DATA_ROOT} and "
            f"implement frame/camera/instruction loading (see data/README.md)."
        )


class CalvinTrajectory(_NotImplementedTrajectory):
    """TODO: load a CALVIN (ABCD) episode from ``data/calvin/...``."""


class LiberoTrajectory(_NotImplementedTrajectory):
    """TODO: load a LIBERO episode (all 4 suites) from ``data/libero/...``."""


class BridgeV2Trajectory(_NotImplementedTrajectory):
    """TODO: load a static-camera BridgeData-V2 episode from ``data/bridge_v2/...``."""

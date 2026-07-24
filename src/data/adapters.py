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

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# Root of the datasets tree (the repo `data/` symlink -> /workspace/datasets/gd_4d_data).
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
LIBERO_ROOT = DATA_ROOT / "libero" / "hdf5"   # data/libero -> /workspace/datasets/libero


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


class LiberoTrajectory:
    """One LIBERO demonstration exposed as a :class:`Trajectory`.

    LIBERO stores demos in robomimic-format HDF5: ``data/demo_<k>/obs/<cam>_rgb`` is
    ``[T, H, W, 3]`` uint8, and the task language lives in ``data.attrs['problem_info']``.
    Images use the OpenGL convention (bottom-up), so we flip vertically to a natural
    top-down frame (``macros_image_convention == 'opengl'``).

    Camera: the third-person ``agentview`` camera is static. We supply **nominal** pinhole
    intrinsics from the render vertical FOV (``fovy_deg``, robosuite default 45°); the
    extrinsic is an identity placeholder. The 4D backbone infers camera itself and S1/S2 do
    not consume these, so exact calibration is deferred to plan §3 / Decision D2 — refine
    before the policy (M3) needs metric camera geometry.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        demo_index: int = 0,
        cam: str = "agentview_rgb",
        fovy_deg: float = 45.0,
        flip_vertical: bool = True,
    ) -> None:
        import h5py

        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"LIBERO hdf5 not found: {self.path}")
        with h5py.File(self.path, "r") as f:
            data = f["data"]
            demos = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
            if not 0 <= demo_index < len(demos):
                raise IndexError(f"demo_index {demo_index} out of range (0..{len(demos) - 1})")
            self._key = demos[demo_index]
            obs = data[self._key]["obs"]
            if cam not in obs:
                raise KeyError(f"camera '{cam}' not in obs; available: {list(obs.keys())}")
            rgb = np.asarray(obs[cam][:], dtype=np.uint8)          # [T, H, W, 3]
            self._instr = self._extract_instruction(data, self.path)
        if flip_vertical:
            rgb = rgb[:, ::-1]                                       # OpenGL bottom-up -> top-down
        self._frames = np.ascontiguousarray(rgb)
        self.episode_id = f"libero/{self.path.stem}/{self._key}"

        T, H, W, _ = self._frames.shape
        fy = (H / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)     # square pixels (square render)
        self._K = np.array([[fy, 0.0, W / 2.0],
                            [0.0, fy, H / 2.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self._E = np.eye(4, dtype=np.float64)                       # placeholder (see class docstring)

    @staticmethod
    def _extract_instruction(data_group, path: Path) -> str:
        info = data_group.attrs.get("problem_info")
        if info is not None:
            try:
                lang = json.loads(info).get("language_instruction")
                if lang:
                    return str(lang).strip()
            except (json.JSONDecodeError, TypeError):
                pass
        # fallback: parse the filename ("..._demo.hdf5" -> spoken instruction)
        stem = path.stem[:-5] if path.stem.endswith("_demo") else path.stem
        return stem.replace("_", " ").strip()

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def frame(self, i: int) -> np.ndarray:
        return self._frames[i]

    def intrinsics(self) -> np.ndarray:
        return self._K

    def extrinsics(self) -> np.ndarray:
        return self._E

    def instruction(self) -> str:
        return self._instr


def iter_libero_episodes(
    suite: str = "libero_spatial",
    cam: str = "agentview_rgb",
    max_files: int | None = None,
    max_demos_per_file: int | None = None,
    root: Path | None = None,
) -> Iterator[LiberoTrajectory]:
    """Yield :class:`LiberoTrajectory` objects across a LIBERO suite (for S1/S2 sweeps).

    ``max_files`` / ``max_demos_per_file`` cap the sweep for quick runs (``None`` = all).
    """
    import h5py

    suite_dir = (root or LIBERO_ROOT) / suite
    files = sorted(suite_dir.glob("*.hdf5"))
    if max_files is not None:
        files = files[:max_files]
    for hp in files:
        with h5py.File(hp, "r") as f:
            n = len(f["data"].keys())
        n = n if max_demos_per_file is None else min(n, max_demos_per_file)
        for di in range(n):
            yield LiberoTrajectory(hp, demo_index=di, cam=cam)


class BridgeV2Trajectory(_NotImplementedTrajectory):
    """TODO: load a static-camera BridgeData-V2 episode from ``data/bridge_v2/...``."""

"""LIBERO goal-pair dataset for training a SuSIE-style goal-image dreamer.

Each sample is ``(obs_t, goal_{t+offset}, instruction)`` drawn from a LIBERO
demonstration. We use the third-person ``agentview_rgb`` camera (the scene the
dreamer must imagine forward), vertical-flipped to the natural top-down frame
(OpenGL convention, matching ``src/data/adapters.py``).

Frames are preloaded into RAM (libero_spatial ~= 2.4 GB uint8) so the training
loop samples random (demo, t, offset) triples with no disk I/O.

Split is *by demo index within each task* (not by task): the dreamer stays in
the task distribution but is evaluated on unseen demonstrations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

LIBERO_ROOT = Path("/workspace/datasets/libero/hdf5")


def _load_task(hp: Path, cam: str = "agentview_rgb"):
    """Return (list[np.uint8 [T,H,W,3]] per demo, instruction str)."""
    import h5py

    demos_frames = []
    with h5py.File(hp, "r") as f:
        data = f["data"]
        info = data.attrs.get("problem_info")
        instr = None
        if info is not None:
            try:
                instr = json.loads(info).get("language_instruction")
            except (json.JSONDecodeError, TypeError):
                instr = None
        if not instr:
            stem = hp.stem[:-5] if hp.stem.endswith("_demo") else hp.stem
            instr = stem.replace("_", " ").strip()
        keys = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
        for k in keys:
            rgb = np.asarray(data[k]["obs"][cam][:], dtype=np.uint8)  # [T,H,W,3]
            rgb = rgb[:, ::-1]  # OpenGL bottom-up -> top-down
            demos_frames.append(np.ascontiguousarray(rgb))
    return demos_frames, str(instr).strip()


def load_libero_spatial(cam: str = "agentview_rgb", root: Path = LIBERO_ROOT):
    """Preload the whole libero_spatial suite.

    Returns list of dicts: {task, instruction, demos: list[np.uint8 [T,H,W,3]]}.
    """
    suite_dir = root / "libero_spatial"
    files = sorted(suite_dir.glob("*.hdf5"))
    tasks = []
    for hp in files:
        demos, instr = _load_task(hp, cam)
        tasks.append({"task": hp.stem, "instruction": instr, "demos": demos})
    return tasks


def _to_tensor(img_uint8: np.ndarray, size: int) -> torch.Tensor:
    """[H,W,3] uint8 -> [3,size,size] float in [-1,1]."""
    im = Image.fromarray(img_uint8).resize((size, size), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class LiberoGoalPairs(Dataset):
    """Random (obs_t, goal_{t+offset}, instruction) triples from libero_spatial.

    Args:
        split: "train" or "val".
        val_demos_per_task: last N demos of each task's 50 are held out for val.
        offset_min/offset_max: subgoal horizon (frames ahead) sampled per item.
        size: output resolution (default 256, matches the 4D backbone).
        length: nominal dataset length (epoch size); items are sampled on the fly.
    """

    def __init__(
        self,
        tasks: list | None = None,
        split: str = "train",
        val_demos_per_task: int = 5,
        offset_min: int = 15,
        offset_max: int = 30,
        size: int = 256,
        length: int = 20000,
        seed: int = 0,
    ) -> None:
        self.tasks = tasks if tasks is not None else load_libero_spatial()
        self.split = split
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.size = size
        self.length = length
        self._rng = np.random.default_rng(seed)

        # Build the index of (task_idx, demo_idx) usable in this split.
        self.pairs = []
        for ti, t in enumerate(self.tasks):
            n = len(t["demos"])
            train_n = n - val_demos_per_task
            demo_range = range(train_n) if split == "train" else range(train_n, n)
            for di in demo_range:
                self.pairs.append((ti, di))

    def __len__(self) -> int:
        return self.length if self.split == "train" else len(self.pairs)

    def _sample_triple(self, ti: int, di: int, rng: np.random.Generator):
        frames = self.tasks[ti]["demos"][di]
        T = frames.shape[0]
        offset = int(rng.integers(self.offset_min, self.offset_max + 1))
        # t must leave room for a positive offset; if the demo is short, clamp.
        t_max = max(0, T - 1 - self.offset_min)
        t = int(rng.integers(0, t_max + 1)) if t_max > 0 else 0
        g = min(t + offset, T - 1)
        return frames[t], frames[g]

    def __getitem__(self, idx: int):
        if self.split == "train":
            ti, di = self.pairs[int(self._rng.integers(0, len(self.pairs)))]
            rng = self._rng
        else:
            ti, di = self.pairs[idx % len(self.pairs)]
            rng = np.random.default_rng(1000 + idx)  # deterministic val
        obs_u8, goal_u8 = self._sample_triple(ti, di, rng)
        return {
            "obs": _to_tensor(obs_u8, self.size),
            "goal": _to_tensor(goal_u8, self.size),
            "instruction": self.tasks[ti]["instruction"],
        }


def val_eval_items(tasks: list, n_per_task: int = 2, val_demos_per_task: int = 5,
                   offset: int = 25, size: int = 256, seed: int = 7):
    """Deterministic held-out (obs, real_goal, instruction) list for generation/eval.

    Uses a fixed obs frame ~1/3 into each held-out demo and a real goal `offset`
    frames ahead, so the same items reproduce across runs.
    """
    rng = np.random.default_rng(seed)
    items = []
    for t in tasks:
        n = len(t["demos"])
        val_ids = list(range(n - val_demos_per_task, n))
        chosen = val_ids[:n_per_task]
        for di in chosen:
            frames = t["demos"][di]
            T = frames.shape[0]
            t0 = T // 3
            g = min(t0 + offset, T - 1)
            items.append({
                "task": t["task"],
                "instruction": t["instruction"],
                "obs_u8": frames[t0],
                "goal_u8": frames[g],
                "obs": _to_tensor(frames[t0], size),
                "goal": _to_tensor(frames[g], size),
                "t0": t0, "g": g, "demo": di,
            })
    return items


if __name__ == "__main__":
    tasks = load_libero_spatial()
    print(f"tasks: {len(tasks)}")
    for t in tasks[:3]:
        print(f"  {t['task'][:50]:50s} demos={len(t['demos'])} instr='{t['instruction']}'")
    ds = LiberoGoalPairs(tasks, split="train")
    print(f"train pairs (task,demo): {len(ds.pairs)}  epoch len: {len(ds)}")
    vds = LiberoGoalPairs(tasks, split="val")
    print(f"val pairs: {len(vds.pairs)}")
    s = ds[0]
    print(f"sample obs {tuple(s['obs'].shape)} range [{s['obs'].min():.2f},{s['obs'].max():.2f}] "
          f"goal {tuple(s['goal'].shape)} instr='{s['instruction']}'")
    ev = val_eval_items(tasks)
    print(f"val_eval_items: {len(ev)}")

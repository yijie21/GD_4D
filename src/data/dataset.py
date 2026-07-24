"""A torch-agnostic map-style dataset over training tuples.

Precomputes the flat index of ``(trajectory, TupleSpec)`` pairs across a collection of
trajectories so it can be wrapped directly by a ``torch.utils.data.DataLoader`` (it only
needs ``__len__`` / ``__getitem__`` — no hard torch dependency here).
"""
from __future__ import annotations

from collections.abc import Sequence

from .schema import TrainingTuple, TupleSpec, WindowConfig
from .trajectory import Trajectory
from .windowing import build_tuple, plan_tuples


class TupleDataset:
    """Map-style dataset over all S1 tuples in ``trajectories``.

    Example
    -------
    >>> from torch.utils.data import DataLoader           # doctest: +SKIP
    >>> ds = TupleDataset(trajectories, WindowConfig())   # doctest: +SKIP
    >>> loader = DataLoader(ds, batch_size=4, collate_fn=list)  # doctest: +SKIP
    """

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        cfg: WindowConfig = WindowConfig(),
    ) -> None:
        self.trajectories: list[Trajectory] = list(trajectories)
        self.cfg = cfg
        self._index: list[tuple[int, TupleSpec]] = []
        for ti, tr in enumerate(self.trajectories):
            for spec in plan_tuples(tr.episode_id, len(tr), cfg):
                self._index.append((ti, spec))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> TrainingTuple:
        traj_idx, spec = self._index[i]
        return build_tuple(self.trajectories[traj_idx], spec)

    @property
    def specs(self) -> list[TupleSpec]:
        """All index-level specs (no pixels) — handy for stats / debugging."""
        return [spec for _, spec in self._index]

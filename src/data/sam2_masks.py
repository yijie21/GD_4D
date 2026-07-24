"""SAM 2 object-mask provider for S2 corruptions.

Thin wrapper over SAM2AutomaticMaskGenerator that returns clean per-object boolean masks (dropping
the huge background/table masks and tiny specks). Heavy imports (`torch`, `sam2`) live inside the
methods so importing :mod:`data` stays cheap and dependency-free.

Weights + config: see `checkpoints/README.md` (SAM 2 row) and `third_party/sam2`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = _REPO / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt"
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"   # resolved on the sam2 package's hydra search path


class Sam2MaskGenerator:
    def __init__(self, ckpt: str | Path | None = None, cfg: str = DEFAULT_CFG, device: str = "cuda",
                 points_per_side: int = 24, pred_iou_thresh: float = 0.8,
                 stability_score_thresh: float = 0.9) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        ckpt = str(ckpt or DEFAULT_CKPT)
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {ckpt} (see checkpoints/README.md)")
        self.device = device
        self._model = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
        self._mg = SAM2AutomaticMaskGenerator(
            self._model, points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh,
        )

    def object_masks(self, frame_uint8: np.ndarray, min_area_frac: float = 0.003,
                     max_area_frac: float = 0.15, max_masks: int = 12,
                     max_span: float = 0.9) -> list[np.ndarray]:
        """Return up to ``max_masks`` boolean **foreground-object** masks (largest first).

        Filters: keep area fraction in ``[min_area_frac, max_area_frac]`` (drops specks and the
        table/floor), and drop scene-spanning masks whose bbox exceeds ``max_span`` of *both* width
        and height (background). Raise ``max_area_frac`` if you want large objects (arm) too.
        """
        import torch

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            raw = self._mg.generate(np.ascontiguousarray(frame_uint8))
        H, W = frame_uint8.shape[:2]
        area = float(H * W)
        masks = []
        for m in sorted(raw, key=lambda x: -x["area"]):
            frac = m["area"] / area
            if frac < min_area_frac or frac > max_area_frac:
                continue
            seg = np.asarray(m["segmentation"], bool)
            ys, xs = np.where(seg)
            if (ys.max() - ys.min() + 1) > max_span * H and (xs.max() - xs.min() + 1) > max_span * W:
                continue                                          # spans the whole scene → background
            masks.append(seg)
        return masks[:max_masks]

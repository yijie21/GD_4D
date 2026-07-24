# S2 · Corrupt & label — cheap corruption generators

Turn a veridical LIBERO goal frame into labeled **"bad dreams"** — the supervised negatives for the
disagreement head (S4). **No diffusion editor**: SAM 2 masks + geometry/compositing only (plan §5.2 —
optimize for geometric *feature coverage*, not photorealism; the head reads geometry, not pixels).

## The four generators — `src/data/corruptions.py`

| Generator | What it does | Region (→ 16×16 label) |
|---|---|---|
| `cut_paste` (remove / relocate / duplicate) | SAM 2 mask + Telea inpaint + copy-paste | object ∪ pasted location |
| `tps_warp` | TPS-style elastic warp localized to a mask | dilated mask |
| `mismatched_frame` | inject a wrong reached-frame (another episode/task) | whole frame |
| `cross_episode_composite` | paste a foreign object onto the frame | pasted mask |

Each returns `(corrupted frame, pixel region, 16×16 patch label)`; a patch is labeled corrupted iff
**≥25%** of its area overlaps the region (`region_to_patch_labels`). Masks come from
`src/data/sam2_masks.py::Sam2MaskGenerator` (foreground-object filter: area ∈ [0.3%, 15%], drops the
scene-spanning table/background) — **or any source**, so the generators are unit-testable without a GPU:

```bash
/workspace/miniconda3/envs/gd4d5090/bin/python -m pytest src/data/tests/test_corruptions.py -q   # 8 tests
```

## Example figures

| File | What it shows |
|---|---|
| `viz/examples_scene1.png`, `viz/examples_scene2.png` | per-generator, two scenes: **original goal → corruption + pixel region (red) → 16×16 patch label (yellow)** |
| `viz/corruptions_demo.png` | one-frame montage of all six variants |

Overlays: **red** = pixel corruption region · **yellow** = 16×16 patch label · **grid** = the 16×16 label cells.

## Reproduce (env: `gd4d5090`)

Prereqs: SAM 2 weights in `checkpoints/sam2/` (see `checkpoints/README.md`) and LIBERO under `data/libero`.

```bash
cd /workspace/code/GD_4D
PY=/workspace/miniconda3/envs/gd4d5090/bin/python
$PY experiments/S2_corruptions/export_examples.py   # -> viz/examples_scene{1,2}.png
$PY experiments/S2_corruptions/demo.py              # -> viz/corruptions_demo.png
```

Runtime ~30–60 s each (SAM 2 large model load + automatic mask generation on a 256×256 frame).

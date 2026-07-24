# S0 · Backbone sanity — is correspondence recovery real?

**Gate before M1 (the bridge).** The entire GD-4D idea rests on one assumption: the frozen
4D backbone can, through its point-query interface `X(u, t_src → t_tgt)`, recover
**geometrically consistent** correspondences with **calibrated** visibility/confidence. Both
GD-4D heads read the four scalars the interface returns (`xyz_3d`, `uv_2d`, `visibility`,
`confidence`), so if recovery is noise, nothing above it is worth building. S0 tests this
empirically with the **released 48CLIP checkpoint**, on a natural clip and a manipulation
clip (our target domain).

## Result — PASS (with one caveat)

| Check | HOT3D (egocentric, moving cam) | LIBERO (manipulation, static cam) | Verdict |
|---|---|---|---|
| **Identity** median / p90 px | 1.85 / 3.46 | 0.91 / 2.10 | ✅ head + coord convention correct |
| **Fwd–bwd cycle** (visible) median px | 2.9–4.6 across horizons to 47f | 1.8–2.2 across horizons to 47f | ✅ correspondences self-consistent to a few px |
| **Visibility** (per-frame visible-frac) | 0.15–1.0, tracks occlusion/return | 0.83–1.0, occludes under the arm | ✅ `v` feature viable |
| **Confidence** (sigmoid) | **saturated ~0.9999** | **saturated ~1.0000** | ⚠️ non-discriminative — see below |
| **Overlay** (eyeball) | points glued to scene through big head motion | static scene locked; arm-occluded pts flip red | ✅ |

**Conclusions**
1. **Correspondences are recoverable.** Identity ~1–2 px and forward–backward cycle ~2–5 px
   (on 256×256) even at a 47-frame horizon. **D0 is validated in substance, not just "the code exists."**
2. **Zero-shot domain transfer to manipulation works** — LIBERO (never trained on) is *cleaner*
   than HOT3D. This de-risks the **frozen-backbone** assumption for our CALVIN/LIBERO/Bridge target.
3. **Visibility is well-calibrated** → the disagreement head's `v` feature and the plan-position
   matching are viable.
4. ⚠️ **Confidence is saturated (~1.0 everywhere)** → the raw `confidence` scalar is **not**
   discriminative on this checkpoint. **Design implication for S4 (disagreement head):** do not
   lean on raw confidence for the `c_bg` feature; derive disagreement primarily from **cycle
   residual `e_cyc`** and **visibility `v`**, and/or compute our own confidence proxy
   (forward–backward residual, multi-path/multi-source consistency). Logged as a plan note.

Raw numbers + overlays are regenerable under `runs/<tag>/` (git-ignored); a committed snapshot
of both runs' `report_*.json` + `overlay_*.png` is kept in [`evidence/`](evidence/) as proof.

## Visuals — *see* the correspondences

`src/eval/s0_visualize_correspondence.py` renders, per clip: **source→target match lines**,
**forward→backward round-trip** (visible points, matching the metric), **forward tracks** (trails),
and an **animated GIF**. Output → `viz/<clip>/` (committed as `viz/**/*.png`; the large `*.gif`
and the generated `viz/gallery.html` are git-ignored — regenerate with the command below).

- **Gallery (published):** https://claude.ai/code/artifact/489245ec-923b-4211-bacc-a77dafd47e8f
- **Regenerate figures + gallery:**
  ```bash
  PY=/workspace/miniconda3/envs/gd4d5090/bin/python; CKPT=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG
  $PY src/eval/s0_visualize_correspondence.py --config $CKPT/model.yaml --ckpt $CKPT/opend4rt.ckpt \
     --video /workspace/datasets/hot3d/rc_input_001964_puzzle_toy/rgb.mp4 --grid 10 \
     --out experiments/S0_backbone_sanity/viz/hot3d_puzzle_toy --tag hot3d
  $PY src/eval/s0_visualize_correspondence.py --config $CKPT/model.yaml --ckpt $CKPT/opend4rt.ckpt \
     --libero-hdf5 <libero.hdf5> --grid 10 \
     --out experiments/S0_backbone_sanity/viz/libero_spatial_demo0 --tag libero
  $PY experiments/S0_backbone_sanity/build_gallery.py   # -> viz/gallery.html (self-contained)
  ```

## Reproduce

Prereqs: the `gd4d5090` conda env (see [`../../env/README.md`](../../env/README.md)) and the
48CLIP checkpoint (see [`../../checkpoints/README.md`](../../checkpoints/README.md)).

```bash
cd /workspace/code/GD_4D
PY=/workspace/miniconda3/envs/gd4d5090/bin/python
CKPT=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG

# (A) natural video — HOT3D egocentric clip
$PY src/eval/s0_backbone_sanity.py \
  --config $CKPT/model.yaml --ckpt $CKPT/opend4rt.ckpt \
  --video /workspace/datasets/hot3d/rc_input_001964_puzzle_toy/rgb.mp4 \
  --num-frames 48 --grid 24 --chunk 4096 \
  --out experiments/S0_backbone_sanity/runs/hot3d_puzzle_toy --tag hot3d_puzzle_toy

# (B) manipulation — LIBERO spatial demo 0 (our target domain)
$PY src/eval/s0_backbone_sanity.py \
  --config $CKPT/model.yaml --ckpt $CKPT/opend4rt.ckpt \
  --libero-hdf5 "/workspace/datasets/libero/hdf5/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5" \
  --demo-index 0 --num-frames 48 --grid 24 --chunk 4096 \
  --out experiments/S0_backbone_sanity/runs/libero_spatial_demo0 --tag libero_spatial_demo0
```

Runtime ~1–2 min each on one RTX 5090 (13 GB fp32 weights + 48-frame encode).

## What the script does (`src/eval/s0_backbone_sanity.py`)

Self-consistency probes (no ground-truth motion needed — any working tracker must pass them):
- **Identity** — `X(u, a→a)` must return `uv ≈ (u,v)`.
- **Fwd–bwd cycle** — `X(u, 0→b)=uv_b`, then `X(uv_b, b→0)=uv'`; `uv' ≈ (u,v)`. This *is* the
  `e_cyc` disagreement feature, so passing here validates that feature directly.
- **Calibration** — per-frame mean confidence & visible-fraction as horizon grows.
- **Overlay** — forward-track a grid; green=visible, red=occluded.

Coordinate conventions used (verified against the vendored code): query `u,v` **normalized**
`= px/(W-1), py/(H-1)`; model `uv_2d` output **normalized** `[0,1]`; `visibility`/`confidence`
are **logits** (sigmoid applied). Model input is 256×256, `clip_frames=48` (single clip, no
stitching for ≤48 frames).

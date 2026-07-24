# Changelog — GD-4D

Human-readable trace of every change to this repo (newest first). **Rule:** every update
appends an entry here in the same change — see `CLAUDE.md` §2.5. Complements `git log`.

Format: `YYYY-MM-DD · <area>` — what changed, why, files/dirs, git commit (short hash).

---

## 2026-07-24

- **S4 disagreement de-risk (pre-M1): the mechanism works, and the feature must be relative.** _(HEAD — this change)_
  - `src/eval/s4_disagreement_derisk.py`: native-encode `[obs…goal]` clips through the frozen backbone
    (no bridge), per-patch cycle error on the goal, AUROC vs S2 label. Reuses S0 helpers + the generators.
  - Result (`experiments/S4_disagreement_derisk/`): **cycle-error DELTA(corrupt−clean) AUROC 0.78,
    visibility 0.76, absolute cycle 0.59** (arm-motion confound). Disagreement localizes corruptions →
    **GO on M1**; feature must be **relative (clean→injected delta) + visibility**, not absolute — which
    is exactly what the bridge's teacher/student distillation provides (validates the bridge design).
  - README + committed evidence figure (per the §2.6 visual-outputs rule).
  - Files: `src/eval/s4_disagreement_derisk.py`, `experiments/S4_disagreement_derisk/`, `GD-4D_implementation_plan.md`.

- **S2 example figures + "document visual outputs" rule.** `68f09cb`
  - `experiments/S2_corruptions/export_examples.py` → `viz/examples_scene{1,2}.png`: per-generator
    `original → corruption+region → 16×16 label`, two LIBERO scenes with varied objects.
  - `experiments/S2_corruptions/README.md`: generators, overlays legend, and **verbatim reproduce commands**.
  - `CLAUDE.md` §2.6: added a **MANDATORY visual-outputs rule** — any exported figure/montage/gallery/GIF
    must be recorded in a README/doc (exact command + output path + env) in the same change, with the
    generating script kept in the repo.
  - Files: `experiments/S2_corruptions/{export_examples.py,README.md,viz/examples_scene{1,2}.png}`, `CLAUDE.md`.

- **S2: implemented the 4 cheap corruption generators + SAM 2 mask provider.** `e1e71a2`
  - `src/data/corruptions.py`: `cut_paste` (remove/relocate/duplicate via mask + inpaint + copy-paste),
    `tps_warp` (TPS-style elastic warp localized to a mask), `mismatched_frame`, `cross_episode_composite`,
    and `region_to_patch_labels` (≥25%-overlap → 16×16 label). Generators take a mask from any source →
    testable without SAM 2/GPU.
  - `src/data/sam2_masks.py`: `Sam2MaskGenerator` wrapping SAM2AutomaticMaskGenerator with a
    foreground-object filter (area ∈ [0.3%, 15%], drop scene-spanning background — the table was the
    largest raw mask). Heavy imports kept inside methods.
  - Tests: `src/data/tests/test_corruptions.py` (8, pure-geometry) → **23 total passing**.
  - Demo: `experiments/S2_corruptions/demo.py` on real LIBERO goal frames → committed montage
    (`viz/corruptions_demo.png`): region ↔ patch labels align across all 4 types; object corruptions 34–54
    patches, mismatch 256. Files: `src/data/{corruptions.py,sam2_masks.py,__init__.py,tests/test_corruptions.py}`,
    `experiments/S2_corruptions/`, `GD-4D_implementation_plan.md`.

- **S2 setup: pulled SAM 2 (masks) + reshaped S2 to cheap negatives (no diffusion editor).** `3d1005b`
  - Vendored `facebookresearch/sam2` @ `2b90b9f` into `third_party/sam2/` (`.git` removed); `pip install -e`
    into `gd4d5090` (**`--no-build-isolation` + `TMPDIR=/workspace`** — build isolation was re-downloading
    torch+CUDA onto the ~2 GB overlay `/` and ran out of space). Weights `sam2.1_hiera_large.pt` (857 MB,
    Apache-2.0, sha256 `2647878d…`) → `checkpoints/sam2/`. **Verified: 23 clean per-object masks on a LIBERO goal frame.**
  - Provenance: `third_party/README.md`, `checkpoints/README.md`; env note + refreshed lock (`env/`).
  - **Plan S2 reshaped** (design doc §M0/S2): 4 **cheap** generators (cut-paste, TPS warp, mismatched-frame,
    cross-episode composite) — SAM 2 masks only, no InstructPix2Pix; realistic dreamer+DINOv3 negatives deferred
    to the S4 OOD gate (§5.2). Also recorded S4=supervised (§5.2).
  - Files: `third_party/{sam2/,README.md}`, `checkpoints/{sam2/,README.md}`, `env/requirements-gd4d5090*.txt`, `GD-4D_implementation_plan.md`.

- **M0/S1 finish: real LIBERO dataset adapter — S1 now builds tuples from real pixels.** `90770d7`
  - Implemented `src/data/adapters.py::LiberoTrajectory` (+ `iter_libero_episodes`): reads robomimic-format
    LIBERO HDF5, corrects the OpenGL vertical flip, pulls the instruction from `data.attrs['problem_info']`,
    supplies nominal pinhole intrinsics (fovy 45°) + placeholder extrinsic (backbone infers camera; exact
    calibration deferred to §3/D2). Previously all real adapters were `NotImplementedError` stubs — S1 had
    only run on `SyntheticTrajectory`.
  - Exposed LIBERO via the `data/libero` symlink (→ `/workspace/datasets/libero`); logged in `data/README.md`.
  - Tests: added `src/data/tests/test_adapters_libero.py` (3 skip-guarded real-data tests) → **15 passing**.
    End-to-end: one 98-frame episode → 18 tuples; 3-file×5-demo sweep → 343 tuples.
  - This unblocks **S2** (which corrupts real reached-frames). Files: `src/data/{adapters.py,__init__.py,tests/test_adapters_libero.py}`, `data/README.md`, `GD-4D_implementation_plan.md`.

- **S0 correspondence visuals: match-lines / cycle / trails / animated tracks + published gallery.** `f094718`
  - Added `src/eval/s0_visualize_correspondence.py` (reuses the S0 loaders) — renders source→target
    match lines, forward→backward round-trip (visible-only, matching the metric), forward-track trails,
    and animated GIFs, for HOT3D + LIBERO. Cycle figures filter to visible points so they can't
    misrepresent (all-points median is dominated by out-of-frame points).
  - Added `experiments/S0_backbone_sanity/build_gallery.py` → self-contained `viz/gallery.html`
    (data-URI images), published as an Artifact:
    https://claude.ai/code/artifact/489245ec-923b-4211-bacc-a77dafd47e8f
  - `.gitignore`: commit `viz/**/*.png` evidence; ignore large regenerable `*.gif` + generated `gallery.html`.
  - Files: `src/eval/s0_visualize_correspondence.py`, `experiments/S0_backbone_sanity/{build_gallery.py,README.md,viz/**}`, `.gitignore`.

- **S0 backbone-sanity: downloaded 48CLIP weights, built the `gd4d5090` env, verified correspondence recovery.** `28c2f17`
  - **Env:** created standalone conda env **`gd4d5090`** (Python 3.10). Installed **cu128** torch
    (`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`) — deviating from OpenD4RT's `cu124` pin, which
    does NOT run on the RTX 5090 (Blackwell sm_120: "no kernel image"). Verified a real CUDA kernel runs.
    Docs: `env/README.md`, `env/requirements-gd4d5090.txt`, `env/requirements-gd4d5090.lock.txt`.
  - **Weights:** downloaded `OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt` (13 GB, Apache-2.0,
    sha256 `65936f72…`) from HF `Lijiaxin0111/OpenD4RT` into `checkpoints/`. Provenance in `checkpoints/README.md`.
  - **Spike:** added `src/eval/s0_backbone_sanity.py` (identity / fwd–bwd cycle / calibration / overlay,
    reusing the vendored query machinery). Ran on a HOT3D natural clip and a LIBERO manipulation clip.
    **Result: PASS** — identity ~1–2 px, cycle ~2–5 px, well-calibrated visibility, zero-shot on manipulation.
    ⚠️ raw `confidence` saturated (~1.0) → recorded as a design note for S4 (disagreement head).
    Records + committed evidence: `experiments/S0_backbone_sanity/`.
  - **Rule:** added `CLAUDE.md` §2.6 (reproducibility) + `env/` and `experiments/` to the directory table;
    updated §3 (D0 resolved & validated). Files: `CLAUDE.md`, `README.md`, `checkpoints/README.md`,
    `env/*`, `experiments/S0_backbone_sanity/*`, `src/eval/s0_backbone_sanity.py`, `GD-4D_implementation_plan.md`.

- **Vendored OpenD4RT backbone + added change-tracing rule.** `fc9a7ac`
  - Copied `https://github.com/Lijiaxin0111/Open-d4rt.git` @ `bead824` into `third_party/Open-d4rt/`
    (`.git` removed; contents git-ignored). This **resolves Decision D0** — GD-4D's frozen 4D
    backbone (inventory #1) is OpenD4RT. Weights still to download from HF into `checkpoints/`.
  - Recorded provenance in `third_party/README.md`.
  - Added **change-tracing rule** to `CLAUDE.md` §2.5; created this `CHANGELOG.md` and a root `README.md`.
  - Files: `CLAUDE.md`, `CHANGELOG.md`, `README.md`, `third_party/README.md` (+ vendored `third_party/Open-d4rt/`, untracked).

- **feat(M0/S1): sliding-window training-tuple builder + tests.** `8d2da49`
  - Implemented S1 (plan §M0/S1) in `src/data/`: `schema.py` (`WindowConfig` h=8/n=20/stride=4,
    `TupleSpec`, `TrainingTuple`), `windowing.py` (`plan_tuples` + `build_tuple(s)`),
    `trajectory.py` (`Trajectory` protocol + `SyntheticTrajectory`), `dataset.py` (`TupleDataset`),
    `adapters.py` (CALVIN/LIBERO/BridgeData-V2 stubs).
  - Added `pyproject.toml` (pytest `pythonpath=src`), `src/configs/data.yaml`, `src/data/README.md`.
  - Tests: 12 passing (`src/data/tests/test_windowing.py`).

- **chore: scaffold GD-4D repo (rules + directory layout).** `8d0eafe`
  - `git init` at `/workspace/code/GD_4D`. `CLAUDE.md` rules; `src/` skeleton;
    `third_party/` + `checkpoints/` provenance logs; `.gitignore`; `data/` symlink →
    `/workspace/datasets/gd_4d_data`.

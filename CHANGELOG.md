# Changelog — GD-4D

Human-readable trace of every change to this repo (newest first). **Rule:** every update
appends an entry here in the same change — see `CLAUDE.md` §2.5. Complements `git log`.

Format: `YYYY-MM-DD · <area>` — what changed, why, files/dirs, git commit (short hash).

---

## 2026-07-24

- **✅ M4: learned critic is VIABLE — the disagreement signal is salvageable (as a critic, not geometry).** _(HEAD — this change)_
  - After M1/M2/M3 falsified the *geometric* `Dₜ`, tested option A: a small learned critic over DINOv2
    semantic features of `(o_t, goal)` (`critic_derisk.py`).
  - **Q1 capability** (MLP probe, real-frame negatives, no obs/real-fake shortcut): diff_task AUROC
    **0.976**, same_task **0.833** — far above geometry (~0.6) and cosine (0.81/0.61). A linear probe
    scored below-chance (linearity artifact; cosine needs the Δ-norm — the MLP fixes it).
  - **Q2 OOD gate** (critic trained ONLY on real frames, applied to M2 generated dreams): dream-fits-
    this-scene **AUROC 0.779** (both sides generated → clean transfer); real goal preferred over the
    dream 19/20.
  - **⇒ `Dₜ` should be a LEARNED critic on appearance/semantic features — exactly the plan's
    S4-supervised head on D1 DINOv3 features — NOT a geometric read-out of the frozen backbone.** The
    backbone's validated role narrows to dream-conditioning for the policy. Open: within-scene
    fine-grained wrongness untestable on homogeneous libero_spatial (needs libero_object/goal/10 +
    patch features).
  - Recorded in plan progress log + memory. Reconciles the M1/M2/M3 negatives: geometric-freebie dead,
    supervised-appearance-critic path alive.
  - Files: `experiments/M4_learned_critic/**`, `GD-4D_implementation_plan.md`.

- **⭐⭐⭐ M3: cross-consistency de-risk falsifies the geometric-disagreement premise (3rd convergent negative).** `68e70df`
  - **Direction #2** (chosen): test reference-free signals that check the goal against the OBSERVED
    WINDOW rather than itself — motion-continuation, background-rigidity, correspondence-completeness,
    plus a self-cycle baseline. Obs window spaced at the goal horizon so motion extrapolation is well
    defined (`derisk.py`). Negatives: `wrong_demo` (same task, plausible wrong future — same scene),
    `wrong_task`, and `shuffled`/`blank` diagnostics.
  - **Result:** the functional signals (S_bg, S_selfcyc) hit **AUROC ≈ 1.0 on geometrically-destroyed
    goals** (shuffled/blank) — proving they are sensitive — but stay at **chance (~0.60 @H=10, ~0.40
    @H=20) on `wrong_demo`**, the realistic plausible-but-wrong future. **Cross-consistency gave no
    advantage over the self-consistency M1/M2 already showed is inadequate.** `S_incomplete` is
    non-functional (visibility saturated — the S0 problem); `S_motion` is noise at long horizon.
  - **⇒ Reference-free GEOMETRIC consistency (self OR cross) over the frozen backbone detects only
    broken images, not the semantically-wrong-but-coherent dreams a real dreamer produces.** Three
    de-risks (M1/M2/M3) converge and **falsify the core premise** that the frozen 4D backbone supplies
    the disagreement `Dₜ` via geometric consistency. Dₜ must be **learned** (semantic/dynamics critic)
    or the mechanism **rethought** (dreamer self-uncertainty; policy uncertainty; drop the gate).
  - Recorded in plan progress log + memory (`disagreement-signal-constraint`). **Strategic decision forced.**
  - Files: `experiments/M3_cross_consistency/**`, `GD-4D_implementation_plan.md`.

- **⭐⭐ M2: a real in-domain dreamer confirms the frozen backbone's self-consistency is BLIND to real dream errors.** `b2fecf0`
  - **Direction #1** (chosen): pull a real robot goal-dreamer, characterize its failures. Skipped native
    SuSIE (JAX/Flax-only + real-world WidowX domain gap, Blackwell-risky) and ran **SuSIE's recipe
    in-domain**: fine-tuned `timbrooks/instruct-pix2pix` on libero_spatial goal pairs (`data.py`,
    450/50 demo split; `train.py`, 10k steps, GPU0, **loss 0.064→0.025**).
  - The dreamer does **real scene-preserving goal edits** (unlike raw IP2P). Failures: global appearance
    drift (tone/vignette) + coherent-but-wrong arm/object pose (`characterize.py` → `dream_failure_gallery.png`).
  - **⭐ Decisive test** (`s4_on_dreams.py`, 20 items ~5k patches, replays the S4 machinery on generated
    dreams vs real goals): confound-free **cycle-error DELTA AUROC = 0.51 (chance)**; abs cycle 0.86 /
    invisibility 0.87 are the **arm-tracking-hardness confound** (the label lives on the intrinsically
    hard-to-track arm; the real-goal clip is the built-in correct-dream control, and delta≈0 proves abs
    can't discriminate correct-vs-wrong).
  - **Generalization:** S2 synthetic corruptions (cut-paste/tps) create a **local geometric inconsistency**
    that S4 delta caught (0.78); real generative dreams are **globally coherent but semantically wrong** →
    no such inconsistency → delta fails. **⇒ S2 corruptions are not representative of real dreamer failures;
    a detector built on backbone self-consistency (abs OR relative-to-synthetic) will not transfer.** The
    disagreement signal must be **cross-consistency** (dream vs what multiple observed frames imply) or a
    **learned semantic critic** over dream+obs — not the frozen backbone's own cycle/visibility.
  - Recorded in plan progress log + memory (`disagreement-signal-constraint`). **Core-mechanism decision forced.**
  - Files: `experiments/M2_libero_dreamer/**`, `GD-4D_implementation_plan.md`.

- **⭐ Premise test: generated dreams are NOT detectable by the backbone's self-consistency (reshapes the disagreement mechanism).** `2c71ea8`
  - Installed InstructPix2Pix (diffusers/transformers/accelerate; weights → HF cache on /workspace).
    `experiments/M1_generated_dream_premise/premise.py`: generate dreams from `o_t`+instruction, native-encode
    `[obs,dream,dream]`, measure cycle error + visibility for real / hallucinated / preserved dreams.
  - **Result:** all three ~identical (cycle 1.83/1.78/1.85 px; visibility ~0.99) — a hallucinated dream is
    self-consistent but wrong. **Cycle consistency ≠ correctness** (same lesson as S0/S4). ⇒ absolute
    cycle/visibility can't detect a wrong dream → the disagreement `Dₜ` must come from **dream↔observed
    cross-consistency**, not the dream's own self-consistency (not yet designed). Also: **IP2P inadequate**
    as a dreamer (hallucinate-or-copy) → need SuSIE/GR-MG.
  - Recorded in plan progress log + memory (`disagreement-signal-constraint`). Core-mechanism decision pending.
  - Files: `experiments/M1_generated_dream_premise/**`, `GD-4D_implementation_plan.md`.

- **M1: full distillation run (bidirectional + all offsets + checkpointing) — machinery complete, gate not yet passed.** `1ea2020`
  - `distill.py` rewritten: bidirectional queries (o_t↔dream), dreams at all offsets j∈{5,10,15,20},
    per-sample `dream_j`, held-out **recall@3px** eval (the acceptance metric). `wrapper.py` gained
    **gradient checkpointing** (`use_checkpoint`) — fixed a GPU0 OOM (grads flow to LoRA through the whole
    1.16 B backbone). GPU0 only.
  - First run (40 train/12 held-out, 150 steps): held-out recall@3px **0.91→0.91 fwd, 0.89→0.90 bwd** —
    **below the 0.95 gate**. Train loss 0.16→0.027 but held-out flat → overfitting + the untrained student
    is already ~0.91 (backbone natively handles obs+dream). Honest result recorded; decision needed
    (scale / refine metric / revisit bridge value). `experiments/M1_bridge_distill/` (recall figure).
  - Files: `src/train/distill.py`, `src/models/bridge/wrapper.py`, `experiments/M1_bridge_distill/README.md`, `GD-4D_implementation_plan.md`.

- **M1: wired imagined-time + dream-token gating into the bridge (identity preserved).** `3dd5fe6`
  - `wrapper.py::encode_video` is now a **bridge-controlled re-implementation of the encoder forward**
    (mirrors the vendored one; **identity smoke still 0.00e0**). In student mode it adds the imagined-time
    embedding to the dream tokens and **gates** each block's LoRA delta to the dream's **last temporal
    patch** (local/global reshape-aware) → observed frames stay bit-identical to the teacher.
  - `distill.py` now encodes the student with `dream_j=n` and trains LoRA + imagined-time. Overfit-8
    residual improved **0.0083 → 0.0068** with both active.
  - Files: `src/models/bridge/wrapper.py`, `src/train/distill.py`, `experiments/M1_bridge_distill/README.md`, `GD-4D_implementation_plan.md`.

- **M1: S3 distillation loop — student→teacher training runs and converges (GPU0, teacher-cached).** `e245ffc`
  - `src/train/distill.py`: teacher = frozen backbone on the full real clip (28f, cached); student = bridge
    on `[obs, dream, dream]` (10f — dream duplicated so the 2-frame temporal patching keeps it). Loss =
    SmoothL1(xyz)+0.1·MSE(conf), Adam on the LoRA params only. Overfit-8-tuples smoke: **loss 0.045→0.008
    (82% down)**, grads flow to LoRA. Records + loss curve in `experiments/M1_bridge_distill/` (per §2.6).
  - Not yet active (next increments): imagined-time wiring into the encoder forward + dream-token gating +
    bidirectional queries + the acceptance-recall gate.
  - Files: `src/train/distill.py`, `experiments/M1_bridge_distill/`, `GD-4D_implementation_plan.md`.

- **M1: bridge (S3) LoRA-injection plumbing — weight-exact identity on the frozen backbone.** `5a11a50`
  - `src/models/bridge/`: `lora.py` (gated, zero-init `LoRALinear` with teacher/student `enabled`),
    `attention.py` (`LoRAAttention.from_mha` — unpacks `nn.MultiheadAttention`'s packed QKV into
    weight-exact Q/K/V/O linears + LoRA), `imagined_time.py` (sinusoidal(j)→MLP), `wrapper.py`
    (`D4RTBridge`: freeze backbone, swap all 40 encoder blocks, add imagined-time, teacher/student modes).
  - Tests: `src/models/bridge/tests/test_bridge.py` (5, incl. exact MHA-match). Smoke:
    `src/eval/m1_bridge_smoke.py` on OpenD4RT → **teacher & zero-init student reproduce the frozen memory
    + query xyz bit-for-bit (max|Δ| = 0.00e0)**. 9.22 M trainable (0.79%).
  - Next increments (M1): dream-token gating, teacher/student clip construction, distillation loop.
  - Files: `src/models/{__init__.py,bridge/**}`, `src/eval/m1_bridge_smoke.py`, `GD-4D_implementation_plan.md`.

- **S4 de-risk: added direct separability visuals (ROC + score distribution + label-overlaid heatmaps).** `954d441`
  - `s4_disagreement_derisk.py` now also emits `viz/s4_derisk_roc.png` (ROC for delta/visibility/absolute)
    and `viz/s4_derisk_distribution.png` (clean vs corrupted delta histogram, n=5041 vs 847 patches, median
    0.00→1.00 px); example heatmaps now overlay the S2 label outline (cyan). Documented in the S4 README.
  - Files: `src/eval/s4_disagreement_derisk.py`, `experiments/S4_disagreement_derisk/{README.md,viz/*}`.

- **S4 disagreement de-risk (pre-M1): the mechanism works, and the feature must be relative.** `6f90b2a`
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

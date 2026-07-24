# GD-4D — project rules & guide

**Grounded Dreaming (GD-4D):** a goal-image-conditioned robot policy built on a **frozen
queryable 4D reconstruction backbone**. A dreamed goal image is injected into the backbone
so that "how wrong is the dream" and "where am I in the plan" become **measured, per-region,
execution-time** quantities that gate the policy.

**This file records the working rules for this codebase. Read it before creating files,
cloning code, or downloading checkpoints.**

---

## 0. Design docs — the source of truth for WHAT to build

Located under `/workspace/research/d4rt/`:

- **Implementation plan & progress tracker** → `GD-4D_implementation_plan.md` (+ `.html`) — **follow this.** It defines milestones **M0–M5** and steps **S1–S9**, with shapes, losses, hyperparameters, acceptance gates, and the backbone/component inventory.
- Idea card → `GD-4D_idea.html` (中文 / English / reviewer).
- Figures → `figures/gallery.html` (fig1–fig9).
- Original D4RT paper (OCR) → `d4rt.pdf_by_PaddleOCR-VL-1.6.md`.

When implementing, work milestone by milestone in the plan and **keep the plan's Progress Log updated** as pieces land.

---

## 1. Directory rules (MANDATORY)

**Working root:** `/workspace/code/GD_4D`

| Folder | Rule |
|--------|------|
| `src/` | **ALL of our own code lives here.** Nothing of ours goes outside `src/`. |
| `third_party/` | **ANY external / cloned repo goes here** (`git clone` or submodule). Do not scatter clones elsewhere. Prefer wrapping vendored code over editing it in place. |
| `checkpoints/` | **ALL downloaded model weights go here.** Never commit weights to git. |
| `env/` | **Environment specs** — the conda env (`gd4d5090`) recreation guide + pinned requirements. |
| `experiments/` | **Experiment records + outputs** (reports, overlays), one folder per experiment, each with a reproducible `README.md`. Not code (code is in `src/`), not weights. |
| `data/` → `/workspace/datasets/gd_4d_data` | **ALL datasets are downloaded to `/workspace/datasets/gd_4d_data`** (outside the code tree), and referenced through the `data/` **symlink** in this repo (`data/` → that folder). Use `data/<dataset>/...` in code; never download datasets into the code tree, and never commit them. |

Do **not** put code, clones, weights, or datasets anywhere else in the tree.

> **Note — `data/` vs `src/data/`:** the top-level `data/` symlink points at the raw **datasets**
> (CALVIN, LIBERO, BridgeData V2, …). `src/data/` is the **data-loading / processing code** (S1 tuples,
> S2 corruptions). They are different things — don't confuse them.
>
> Recreate the symlink if it's missing: `ln -s /workspace/datasets/gd_4d_data /workspace/code/GD_4D/data`

### Intended `src/` layout (from the plan)
```
src/
  data/        # S1 build tuples, S2 corrupt & label
  models/
    backbone/  # frozen 4D reconstruction F   (see D0 below — which backbone is TBD)
    bridge/    # S3 LoRA adapters + imagined-time embedding
    heads/     # S4 disagreement head hψ, S5 plan-position head pω
    policy/    # S6 conditioning tensor + policy head (Diffusion Policy / GR-1)
  train/       # training loops (distillation, head training, BC)
  eval/        # S8 suites, S9 mechanism controls
  configs/     # experiment configs
```

---

## 2. Conventions

- **Python + PyTorch.** Activate the env: `source /venv/main/bin/activate`. Install with `uv pip install <pkg>`.
- **Third-party provenance:** whenever you clone into `third_party/`, append an entry to `third_party/README.md` (repo URL + exact commit hash + why it's used).
- **Checkpoint provenance:** whenever you download weights into `checkpoints/`, append an entry to `checkpoints/README.md` (component # from the inventory + source URL + sha256 + license).
- **Dataset provenance:** whenever you download a dataset into `/workspace/datasets/gd_4d_data/`, append an entry to `data/README.md` (dataset name + source URL + version/split + license).
- **Frozen vs trained:** by design the 4D backbone and the dreamer/SAM are **frozen**; only the LoRA bridge + imagined-time embedding, the two heads, and the policy are **trained**. Keep this boundary explicit in code (e.g., `requires_grad=False` on frozen modules).
- **Configs, not constants:** every hyperparameter in the plan lives in a config, not hard-coded.
- **Git:** `checkpoints/` and `third_party/` contents are git-ignored (see `.gitignore`); only `README.md` + `.gitkeep` are tracked there. The `data/` symlink is git-ignored too (datasets live outside the repo).

---

## 2.5 Change tracing (MANDATORY)

**Every time something updates — code, vendored deps, checkpoints, configs — append an entry
to `CHANGELOG.md` at the repo root, in the same change.** Record the exact change so it can be
traced later:

- **date**, **what changed** (files / dirs), **why**, and the **git commit** (short hash) if committed.
- Keep newest entries at the top. This is a human-readable trace that complements `git log`.

Do not land a change without a corresponding `CHANGELOG.md` entry.

## 2.6 Reproducibility (MANDATORY)

**Every environment, download, and experiment must be reproducible by a third person from the
committed docs alone — no reliance on shell history or undocumented steps.** Concretely:

- **Environment:** the single conda env for this repo is **`gd4d5090`** (built for the RTX 5090 /
  Blackwell — needs **cu128** torch, *not* OpenD4RT's cu124 pin). Any change to it (new package,
  version bump) updates `env/README.md` + `env/requirements-gd4d5090*.txt` in the same change.
- **Downloads** (weights, datasets): record the **exact command**, source URL, **sha256**, and
  license in the relevant provenance log (`checkpoints/README.md`, `data/README.md`,
  `third_party/README.md`) so the artifact can be re-fetched byte-for-byte.
- **Experiments:** each lives under `experiments/<name>/` with a `README.md` that states the
  purpose, the **verbatim commands** to reproduce it, the environment used, and the results
  (with the metric files / outputs alongside). A third person should be able to `cd` in and
  re-run it top to bottom.
- **Visual outputs (MANDATORY):** whenever you produce ANY visual deliverable — a figure, montage,
  gallery, GIF, or overlay (i.e. any time the user says "export/draw/render/visualize/plot" or you
  otherwise emit an image) — record it in the relevant `README.md` / doc **in the same change**: the
  **exact command** that regenerates it, the **output path**, and the **env**. The generating script
  lives in the repo (never a throwaway), so a third person can reproduce the exact image. No undocumented figures.

If a step isn't written down such that someone else can repeat it, it isn't done.

## 3. Backbone status (D0 — RESOLVED & VALIDATED)

**Decision D0 is resolved: the frozen 4D backbone is OpenD4RT** (`third_party/Open-d4rt`,
48CLIP checkpoint in `checkpoints/`). It is not just present — **S0 (`experiments/S0_backbone_sanity/`)
empirically confirmed correspondences are recoverable** (identity ~1–2 px, fwd–bwd cycle ~2–5 px,
well-calibrated visibility) and that recovery **transfers zero-shot to manipulation** (LIBERO).
So M1 (the bridge) is unblocked.

⚠️ **Carry this finding into S4:** the checkpoint's raw `confidence` scalar is **saturated
(~1.0 everywhere)** — non-discriminative. The disagreement head must derive `Dₜ` primarily from
the **cycle residual `e_cyc`** and **visibility `v`** (and/or a computed confidence proxy), not
raw confidence.

---

## 4. Build order (quick reference)

`M0` data (S1 tuples → S2 labels) → `M1` bridge (S3, needs D0) → `M2` heads (S4 hψ, S5 pω)
→ `M3` policy (S6) → `M4` executor (S7) → `M5` eval (S8, S9). See the plan for full specs.

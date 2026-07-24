# Changelog — GD-4D

Human-readable trace of every change to this repo (newest first). **Rule:** every update
appends an entry here in the same change — see `CLAUDE.md` §2.5. Complements `git log`.

Format: `YYYY-MM-DD · <area>` — what changed, why, files/dirs, git commit (short hash).

---

## 2026-07-24

- **Vendored OpenD4RT backbone + added change-tracing rule.** _(HEAD — this change)_
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

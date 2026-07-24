# GD-4D

**Grounded Dreaming (GD-4D):** a goal-image-conditioned robot policy built on a **frozen
queryable 4D reconstruction backbone**. A dreamed goal image is injected into the backbone so
that *"how wrong is the dream"* and *"where am I in the plan"* become **measured, per-region,
execution-time** quantities that gate the policy.

## Repo map

| Path | What |
|------|------|
| `CLAUDE.md` | **Project rules** — directory layout, conventions, change-tracing. Read first. |
| `CHANGELOG.md` | **Change trace** — every update is recorded here (newest first). |
| `src/` | Our code (`data/`, `models/{backbone,bridge,heads,policy}/`, `train/`, `eval/`, `configs/`). |
| `third_party/` | Vendored external repos (e.g. OpenD4RT). Provenance in `third_party/README.md`. |
| `checkpoints/` | Downloaded weights (git-ignored). Provenance in `checkpoints/README.md`. |
| `data/` → `/workspace/datasets/gd_4d_data` | Datasets (symlink; git-ignored). |

Design docs (the spec for *what* to build) live in `/workspace/research/d4rt/`:
`GD-4D_implementation_plan.md` (milestones M0–M5), `GD-4D_idea.html`, `figures/gallery.html`.

## Status

- **M0 · S1 (build tuples)** — ✅ done. **M0 · S2 (corrupt & label)** — next.
- **D0 (backbone)** — resolved: **OpenD4RT** vendored in `third_party/Open-d4rt/` (weights TBD from HF).
- See `CHANGELOG.md` for the full trace and the plan for milestone specs.

## Run the tests

```bash
source /venv/main/bin/activate
cd /workspace/code/GD_4D
pytest            # pythonpath=src, testpaths=src (see pyproject.toml)
```

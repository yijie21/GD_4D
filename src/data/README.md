# src/data — GD-4D data pipeline (Milestone M0)

Implements the data steps of the plan.

## Status
- **S1 · Build training tuples** — ✅ implemented (`schema.py`, `trajectory.py`, `windowing.py`, `dataset.py`).
- **S2 · Corrupt & label** — ☐ next (`corruptions.py`, to add).

## Modules
| File | What it is |
|------|------------|
| `schema.py` | `WindowConfig`, `TupleSpec` (index-level), `TrainingTuple` (materialized). |
| `trajectory.py` | `Trajectory` protocol + `SyntheticTrajectory` (test fixture, no dataset needed). |
| `windowing.py` | `plan_tuples` (pure sliding-window) + `build_tuple(s)` (materialize pixels). |
| `dataset.py` | `TupleDataset` — torch-agnostic map-style dataset over many trajectories. |
| `adapters.py` | Dataset adapters (CALVIN / LIBERO / BridgeData V2) — **stubs**, fill when data lands. |

## S1 windowing rule
For current time `t`, history `h=8`, horizon `n=20`:
```
obs       = [t-h+1 .. t]     # 8 frames, ending at t
waypoints = [t+1   .. t+n]   # 20 frames, the last being the goal
goal      = t+n              # the veridical "perfect dream"
```
Valid `t ∈ [h-1 .. L-1-n]`, stepped by `stride=4` (shortest usable trajectory: `L = h+n = 28`).

## Run the tests
```bash
source /venv/main/bin/activate
cd /workspace/code/GD_4D
pytest                      # uses pyproject.toml (pythonpath=src, testpaths=src)
# or: python src/data/tests/test_windowing.py
```

## Next
Implement S2 (`corruptions.py`): 16×16 patch masks + four corruption generators (SAM + editor),
plus scoring real dreamer outputs. Blocked by Decision **D1** (label source). See the plan §M0/S2.

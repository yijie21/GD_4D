# third_party/

External / cloned repositories used by GD-4D. **Do not scatter clones elsewhere — everything vendored lives here.**
Contents are git-ignored; record every clone below so it is reproducible.

## Log

| Repo (URL) | Commit | Used for | Date |
|------------|--------|----------|------|
| https://github.com/Lijiaxin0111/Open-d4rt.git | `bead824` | **D4RT-style 4D reconstruction backbone** (inventory component #1) — resolves Decision D0. Model at `src/model/` (`d4rt.py`, `encoder.py`, `decoder.py`, `query_embedding.py`, `heads.py`); the queryable interface the M1 bridge wraps. | 2026-07-24 |
| https://github.com/facebookresearch/sam2.git | `2b90b9f` | **SAM 2** (inventory #6) — object masks for S2 corruptions (cut-paste, composite). `pip install -e third_party/sam2` into `gd4d5090`; weights in `checkpoints/sam2/`. Apache-2.0. | 2026-07-24 |

Notes:
- Vendored from local copy `/workspace/code/Open-d4rt` (its `.git` removed here; commit recorded above).
- OpenD4RT = unofficial PyTorch reimpl of D4RT (RHOS Team), Apache-2.0, PyTorch 2.6 / Python 3.10.
- **Weights are NOT bundled** — download from Hugging Face `Lijiaxin0111/OpenD4RT` into `../checkpoints/` (log them in `checkpoints/README.md`).

<!-- Example row:
| https://github.com/facebookresearch/segment-anything | <sha> | SAM masks for S2 corruption | 2026-07-24 |
-->

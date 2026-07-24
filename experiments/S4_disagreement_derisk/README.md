# S4 · Disagreement de-risk — does geometric disagreement separate clean vs. corrupted?

Tested **before** building the bridge (M1) and **before** any big materialize (same de-risk-first
approach as S0). Native-encode a clip `[obs … goal]` through the *frozen* backbone (S2 corruptions are
edits of real frames → they encode fine), compute a **per-patch cycle error** on the goal frame, and
measure AUROC against the S2 patch label.

## Result — GO, with a design refinement

Pooled over 23 localized-corruption tuples (LIBERO), 16×16 patches each:

| Per-patch score | AUROC |
|---|---|
| cycle-error **DELTA** (corrupt − clean) | **0.78** |
| (1 − visibility) on the corrupt clip | **0.76** |
| absolute cycle error on the corrupt clip | 0.59 |

Whole-frame `mismatch` (global): mean cycle **clean 5.28 vs corrupt 4.80** — no rise.

**Conclusions**
1. **Disagreement localizes corruptions** (delta 0.78, visibility 0.76) → the S4 mechanism is real → **build M1**.
2. **Absolute cycle error is confounded** by *legitimate* high motion — the robot arm moves a lot between
   obs and goal, so its cycle error is high whether or not the dream is wrong (→ weak 0.59). You **must**
   measure disagreement **relative to a clean reference.**
3. That relative reference is **exactly what the bridge's teacher–student distillation provides**
   (teacher = frozen `F` on the real frame; student = injected dream). So this de-risk **validates the
   bridge design** and refines S4: features = clean→injected **DELTA** + **visibility**, *not* absolute cycle.
4. `mismatch` (whole-wrong frame) doesn't raise absolute cycle — a wrong frame is still internally
   *self-consistent*. Same lesson: the signal is **relative**, not absolute.

**Caveats.** Native-encode is a proxy (no bridge, no imagined-time embedding; the "clean reference" here
is a re-encoded clean clip, not a true teacher). Small sample (23 tuples, LIBERO only). Expect the real
bridge delta to differ — this establishes the mechanism exists and how to read it, not the final number.

## Reproduce (env `gd4d5090`)

```bash
cd /workspace/code/GD_4D
CKPT=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG
/workspace/miniconda3/envs/gd4d5090/bin/python src/eval/s4_disagreement_derisk.py \
  --config $CKPT/model.yaml --ckpt $CKPT/opend4rt.ckpt --n-files 7 --per-file 4
# -> experiments/S4_disagreement_derisk/report.json + viz/s4_derisk_examples.png
```

Figure `viz/s4_derisk_examples.png`: **corrupted goal | cycle-error rise vs clean | S2 label** — the
delta lights up on the corrupted region (removed / relocated / duplicated object), matching the label.

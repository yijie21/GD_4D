# M4 — Learned-critic de-risk: can a semantic-feature judge do what geometry can't?

**Direction chosen after M3.** M1/M2/M3 falsified the *geometric* disagreement signal: the frozen
backbone's cycle/visibility (self or cross) can't tell a wrong dream from a right one. This tests
option A — a small **learned critic over semantic features** (DINOv2), which is what the disagreement
head `Dₜ` would actually be. (This is exactly the plan's **S4-supervised head on D1 appearance
features** — not a geometric read-out of the backbone.)

## Design

Feature per `(o_t, goal)`: `[emb(o_t), emb(goal), emb(goal) − emb(o_t)]` — DINOv2-base CLS (768·3).
Probe: a small MLP (a linear probe fails — cosine similarity needs the *norm* of the difference,
which a linear model can't compute). Positives and negatives share the SAME `o_t`, so there is no
obs shortcut, and (in Q1) both are real frames, so no real-vs-generated shortcut.

## Q1 — capability (real-frame negatives)

| negative | probe (MLP) AUROC | cosine baseline | geometry (M3) |
|---|---|---|---|
| `diff_task` (goal from a different task — wrong scene) | **0.976** | 0.806 | ~0.75 |
| `same_task` (goal from a different demo, same task) | **0.833** | 0.610 | ~0.60 |

A learned critic on semantic features separates correct from wrong goals **far better than geometry
or cosine** — including the harder within-task case (0.83). The capability geometry lacked is present
in a learned semantic critic. (The linear probe scored *below chance* — an artifact of linearity, not
the features; the MLP fixes it.)

## Q2 — the OOD gate: does it transfer to REAL generated dreams?

The critic trained **only on real frames** is applied to the M2 dreamer's generated dreams:

| test | result |
|---|---|
| dream-fits-this-scene: `(o_i, dream_i)` vs `(o_i, dream_j)` | **AUROC 0.779** |
| real goal scored above the dream for the same `o_t` | 19/20 (0.95) |

The first is the clean transfer number — both sides are *generated* dreams (no real-vs-fake
shortcut), differing only in whether the dream matches the obs's scene. **The real-trained critic
transfers across the real→generated domain gap (0.78).** The 0.95 real-vs-dream preference is
suggestive but partly a domain confound (dreams look "generated").

## Verdict

**The learned critic is viable where the geometric signal was falsified.** Contrast:

| approach | correct-vs-wrong goal | transfers to real dreams |
|---|---|---|
| geometric self/cross-consistency (M1/M2/M3) | ~0.5–0.6 (chance) | no (M2: 0.51) |
| **learned semantic critic (M4)** | **0.83–0.98** | **0.78 (scene-fit)** |

⇒ `Dₜ` should be a **learned critic on appearance/semantic features** (the plan's S4-supervised head
on D1 DINOv3 features), **not** a geometric read-out of the frozen backbone. The backbone's validated
role narrows to **dream-conditioning for the policy** (its real strength), not supplying `Dₜ`.

### Open (honest scope)
- Q2 tests **scene-fit**; **within-scene fine-grained** correctness (a coherent dream with the object
  in the *wrong place*, same scene) was not cleanly testable — libero_spatial's low task diversity
  (all "pick black bowl → place on plate") makes a same-scene *wrong* dream ≈ a correct one. Needs a
  more diverse suite (libero_object/goal/10) or patch-level features to localize the error.
- The critic uses global CLS; **patch features** (localizing a wrong object) are the natural next step
  and align with D1's patch-cosine design.

## Reproduce (env `gd4d5090`, GPU0)
```bash
cd /workspace/code/GD_4D
export HF_HOME=/workspace/huggingface_cache/
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/gd4d5090/bin/python \
    experiments/M4_learned_critic/critic_derisk.py
# -> q1_report.json   (Q1 capability + Q2 transfer; needs M2 dreams/dreams.npz for Q2)
```
No figures exported (numeric de-risk); results are in `q1_report.json` + stdout.

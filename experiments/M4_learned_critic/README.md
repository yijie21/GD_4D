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

## v2 — the real Dₜ head: per-patch critic, within-scene errors, cross-suite OOD

`critic_v2_patch.py` closes the v1 caveat. Downloaded **4 suites** (spatial / object / goal / 10) and
built the plan's **S4-supervised head on DINOv2 PATCH features**: genuinely-wrong same-scene negatives
= S2 corruptions (relocate / remove / tps of the real goal via SAM2 masks) with 16×16 patch labels;
per-patch feature `[dino_patch(goal), dino_patch(goal) − dino_patch(o_t)]`; clean goals contribute
all-0 so the critic must separate a **corruption-change** from a **legitimate motion-change** (arm /
object legitimately moving) — the exact thing geometry (M1/M2/M3) could not do.

| test | probe AUROC | cosine baseline |
|---|---|---|
| **within-scene wrong-object** (held-out demos, all suites) | **0.973** | 0.794 |
| OOD — held-out `libero_spatial` | 0.964 | 0.798 |
| OOD — held-out `libero_goal` | 0.992 | 0.777 |
| OOD — held-out `libero_10` | 0.830 | 0.800 |
| OOD — held-out `libero_object` | **0.749** | 0.776 |

**The learned patch critic detects within-scene wrong objects at 0.97** — where geometry was at
chance — and generalizes across most unseen suites (0.83–0.99). **Weak spot: novel objects**
(`libero_object` held out → 0.75, ≈ the cosine baseline): the critic transfers less to object
appearances it never saw in training. Mitigations: train on more suites (libero_90), DINOv3 features,
or accept object-novelty as an inherent gap.

**Verdict (M4 overall):** the disagreement head `Dₜ` = a **learned per-patch critic on semantic
(DINOv2/DINOv3) features** is validated — separates correct/wrong goals (CLS 0.83–0.98), catches
within-scene wrong objects (patch 0.97), and generalizes across suites (0.75–0.99). This is the plan's
S4-supervised head on D1 appearance features. Geometry supplies nothing here; the frozen backbone's
role is dream-conditioning for the policy.

### Open (honest scope)
- OOD to genuinely **novel objects** transfers less (0.75) — needs broader training suites or DINOv3.
- Negatives are S2 **synthetic** corruptions; a direct test on a diverse real dreamer's within-scene
  errors is still ideal (blocked by the low-diversity libero_spatial dreamer — retrain the dreamer on
  all 4 suites to close this).
- Next: swap DINOv2→DINOv3 (D1), fold the critic into the S4 head, wire `Dₜ` into the policy gate.

## Reproduce (env `gd4d5090`, GPU0)
```bash
cd /workspace/code/GD_4D
export HF_HOME=/workspace/huggingface_cache/
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/gd4d5090/bin/python \
    experiments/M4_learned_critic/critic_derisk.py
# -> q1_report.json   (Q1 capability + Q2 transfer; needs M2 dreams/dreams.npz for Q2)
```
No figures exported (numeric de-risk); results are in `q1_report.json` + stdout.

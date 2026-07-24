# Premise test — is a generated dream OOD for the frozen backbone?

**Question (the bridge's reason to exist):** the bridge (M1) is only needed if native-encode *fails* on
generated dreams. We native-encode `[obs, dream, dream]` and measure the forward→backward **cycle error**
(o_t→dream→o_t) + **visibility** of a 24×24 grid, for three dreams (5 LIBERO tuples).

## Result — the premise is NOT supported (and it's an important negative)

| dream | median cycle err (px) | mean visibility |
|---|---|---|
| **real goal** (in-distribution) | 1.83 | 0.989 |
| **hallucinated** IP2P @ img-guid 1.5 (a totally different scene) | **1.78** | 0.990 |
| **preserved** IP2P @ img-guid 2.75 (~copy of o_t) | 1.85 | 0.992 |

All three are **indistinguishable**. The frozen backbone produces **self-consistent** correspondences for
*any* image — real, hallucinated, or copy.

## Two findings

1. **IP2P is inadequate as the dreamer** (`viz/dream_examples.png`): at low image-guidance it hallucinates a
   new scene (no arm/cabinet); at high image-guidance it just copies `o_t`. It never does a scene-preserving
   *goal* edit (move the bowl to the plate). A robot-goal dreamer (SuSIE / GR-MG) would be needed.
2. **⭐ Cycle consistency ≠ correctness.** A hallucinated dream is self-consistent (cycle ~1.8 px) but
   semantically garbage. So **absolute cycle/visibility cannot detect a wrong dream** — the same lesson as S0
   and the S4 de-risk (where only the *relative* clean→corrupt delta, 0.78 AUROC, separated corruptions).

## Implication (why this matters beyond M1)

The disagreement signal `Dₜ` — the core of GD-4D — is supposed to measure "how wrong is the dream" at
inference, where there is **no clean reference and no real goal**. This test shows the absolute geometric
signals it would read (cycle error, visibility) are **blind to dream correctness**. So the mechanism must
detect a wrong dream by **cross-checking the dream against the observed frames** (does the dream's implied
geometry contradict the observed motion?), not by the dream's own self-consistency. That cross-consistency
detector is **not yet designed or validated** — and is the real open question.

## Reproduce (env `gd4d5090`, GPU0)

```bash
cd /workspace/code/GD_4D
export HF_HOME=/workspace/huggingface_cache/
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/gd4d5090/bin/python experiments/M1_generated_dream_premise/premise.py
# -> viz/premise.png, viz/dream_examples.png   (InstructPix2Pix weights auto-download to HF cache)
```

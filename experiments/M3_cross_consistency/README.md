# M3 — Cross-consistency de-risk: can a REFERENCE-FREE geometric signal flag a wrong dream?

**Direction #2** (chosen after M2): M1+M2 proved the frozen backbone's *self*-consistency (cycle /
visibility of the goal alone) is blind to dream correctness. This tests the alternative the memory
flagged: check the goal against the **observed context** (the obs window), never against itself or a
clean reference — a signal computable at inference from `[obs window, goal]` only.

## Design

Obs window spaced at the goal horizon H: frames at times `[t-2H, t-H, t]` (clip idx 0,1,2) + a
candidate goal (idx 3). One clip-step ≈ H real frames, so the goal sits exactly one velocity-step
after `o_t` and motion extrapolation is well defined. Everything uses the frozen backbone's query
interface; the obs window is always real.

Candidate goals per window (goal at ~`t+H`):
| variant | what | expectation |
|---|---|---|
| `correct` | real future of THIS demo | label 0 |
| `wrong_demo` | real goal of a DIFFERENT demo, **same task** | plausible-but-wrong future, **same scene** — the hard, realistic case |
| `wrong_task` | real goal of a DIFFERENT task | wrong scene — easy |
| `shuffled` | 32-px block-permutation of the correct goal | DIAGNOSTIC: same local content, destroyed geometry |
| `blank` | gray frame | DIAGNOSTIC: no content |

Reference-free signals (from `[obs window, goal]` only):
- **S_motion** — moving points: `‖goal_pos − (o_t + velocity)‖` (does the goal continue the motion?)
- **S_bg** — static points: `‖goal_pos − o_t‖` (did the static scene move?)
- **S_incomplete** — `1 − min(fwd, rev)` grid correspondence completeness (hallucinated / dropped content)
- **S_selfcyc** — BASELINE self-consistency cycle `goal→o_t→goal` (M1/M2 said this is blind)

## Result — reference-free geometric cross-consistency is NOT viable for realistic wrong dreams

AUROC (correct vs each negative). `shuffled`/`blank` are diagnostics that prove the signals are *not*
broken — a working geometric signal MUST fire on destroyed geometry.

**H = 10** (favorable: correspondence is reliable, S_motion fires on shuffled):
| signal | shuffled | blank | wrong_task | **wrong_demo** |
|---|---|---|---|---|
| S_bg | **1.00** | 0.97 | 0.74 | **0.60** |
| S_selfcyc (baseline) | **1.00** | 1.00 | 0.76 | **0.60** |
| S_motion | 0.86 | 0.69 | 0.60 | **0.53** |
| S_incomplete | 0.70 | 0.20 | 0.58 | **0.50** |

**H = 20** (designed horizon):
| signal | shuffled | blank | wrong_task | **wrong_demo** |
|---|---|---|---|---|
| S_bg | **1.00** | 0.99 | 0.39 | **0.37** |
| S_selfcyc | **1.00** | 1.00 | 0.53 | **0.40** |
| S_motion | 0.51 | 0.33 | 0.47 | **0.49** |
| S_incomplete | 0.52 | 0.17 | 0.51 | **0.48** |

### Reading
1. **The signals work** — S_bg and S_selfcyc hit **AUROC ≈ 1.0** on geometrically *destroyed* goals
   (shuffled/blank). They are sensitive; they are not saturated to noise.
2. **They are blind to a plausible wrong future** — `wrong_demo` (a real goal from another demo of the
   same task: same scene, wrong trajectory) stays at **chance (~0.60 at best, ~0.40 at H=20)**.
3. **Cross-consistency gives NO advantage over self-consistency** — S_bg / S_motion never beat the
   S_selfcyc baseline that M1/M2 already showed is inadequate.
4. **Two signals are additionally non-functional**: `S_incomplete` never fires (visibility is
   saturated — the S0 "confidence saturated" problem), and `S_motion` is noise at H=20 (long-range
   correspondence unreliable).

### Conclusion
Reference-free **geometric** consistency over the frozen backbone — self *or* cross — detects only
geometrically **broken** goals (shuffle/blank), not the **semantically-wrong-but-coherent** goals a
real dreamer produces (M2). The backbone finds coherent correspondences for *any* realistic in-domain
image, so geometry cannot answer "is this the CORRECT continuation."

**Three de-risks now converge and falsify the core premise** ("a frozen queryable 4D backbone supplies
the disagreement signal via geometric consistency"):
- **M1 premise**: dream self-consistency identical for real / hallucinated / copied goals.
- **M2**: on a real dreamer, confound-free geometric delta at chance (0.51); abs signals are the
  arm-hardness confound.
- **M3**: reference-free geometric cross-consistency at chance for plausible wrong futures (~0.60),
  no better than self-consistency.

⇒ The disagreement `Dₜ` cannot be a geometric read-out of the frozen backbone. It must be **learned**
(a critic with semantic/dynamics knowledge of what a correct continuation looks like) or the
mechanism rethought (dreamer self-uncertainty; policy uncertainty; or drop the gate).

## Reproduce (env `gd4d5090`, GPU0)
```bash
cd /workspace/code/GD_4D
export HF_HOME=/workspace/huggingface_cache/
BB=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/gd4d5090/bin/python \
    experiments/M3_cross_consistency/derisk.py --config $BB/model.yaml --ckpt $BB/opend4rt.ckpt --H 10
# -> derisk_report.json + derisk_distributions.png   (rerun with --H 20 for the second table)
```

## Visual outputs (per project rule)
| file | produced by |
|---|---|
| `derisk_distributions.png` — per-signal histograms, correct vs wrong_demo vs wrong_task | `derisk.py` |
| `derisk_report.json` — per-signal AUROC + medians for all variants | `derisk.py` |

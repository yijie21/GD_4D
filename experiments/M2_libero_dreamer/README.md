# M2 — A real LIBERO goal-dreamer, and what its failures do to the disagreement signal

**Direction #1** (chosen after the M1 premise test): pull a *real* robot goal-dreamer and
characterize its actual failure modes, instead of relying on the wildly-OOD IP2P scene-swap the
premise test used.

SuSIE (the intended dreamer) is JAX/Flax-only and trained on real-world WidowX (BridgeData) — a
large domain gap from LIBERO sim, and Blackwell-risky. But **SuSIE _is_ "fine-tune InstructPix2Pix
on (current-obs, instruction, future-frame) robot triples."** So we run that exact recipe **on
LIBERO itself**: the dreamer then lives in our deployment distribution, and its failures are the
ones the disagreement head will actually face.

## What we built

1. **Dataset** (`data.py`) — libero_spatial (10 tasks, 500 demos), agentview_rgb, vertical-flipped.
   Split by demo index: 450 train / 50 val demos. Samples `(obs_t, goal_{t+15..30}, instruction)`.
2. **Dreamer** (`train.py`) — init `timbrooks/instruct-pix2pix` (already an 8-channel image-conditioned
   SD1.5 UNet), freeze VAE + text encoder, train the UNet 10k steps on the goal pairs. **Loss
   0.064 → 0.025.** GPU0, bf16, gradient checkpointing, ~46 min.
3. **Generate** (`generate.py`) — 20 held-out items (2 demos × 10 tasks), obs → dreamed goal.
4. **Characterize** (`characterize.py`) — localized-vs-global pixel analysis + gallery.
5. **S4-on-dreams** (`s4_on_dreams.py`) — replay the exact S4 machinery (per-patch cycle error +
   visibility) on the *generated* dreams vs the *real* goal.

## Results

### The recipe works
The fine-tuned dreamer does **scene-preserving goal edits** (unlike raw IP2P, which
hallucinated a whole new scene or copied the input). Dreams reproduce the scene layout (drawer,
plate, ramekin, cookie box) and roughly follow the task's arm/object progression.

### Failure modes (two kinds)
- **Global appearance drift**: tone darkening + corner vignette (systematic SD-finetune artifacts).
- **Localized geometric errors**: arm pose / object placement that is *coherent* (plausible-looking)
  but semantically wrong.

`viz/dream_failure_gallery.png`: `|dream − real_goal|` concentrates on the **arm + manipulated
object** (the dynamic elements). Pixel stats: `excess_conc` 0.48 (above-floor error is concentrated,
not uniform), but `floor` = 34 px (large global tone drift) and `change_iou` = 0.017 (the dream's
_biggest_ changes are the tone/vignette artifacts, not the task region) — pixel space is confounded
by appearance drift, which motivates the geometric test.

### ⭐ The decisive test: backbone geometric self-consistency does NOT catch real dreamer failures

`viz/s4_on_dreams_report.json`, pooled over 20 items (~5k patches), scored against the
`|dream−real_goal|` wrong-patch label:

| signal | pooled AUROC | reading |
|---|---|---|
| absolute cycle error | **0.86** | spurious — arm-hardness confound |
| absolute (1 − visibility) | **0.87** | spurious — arm-hardness confound |
| **cycle-error delta** (dream − real goal) | **0.51** | **chance — the confound-free signal is blind** |
| corr(delta, pixel-error) | ≈ 0.00 | no spatial relation to where the dream is wrong |

**Why the absolute signals are spurious:** the wrong-patch label lives on the arm, and the arm is
intrinsically hard to track (thin, specular, self-occluding) in *any* clip — a correct dream or a
wrong one. So "high cycle error / low visibility" just marks *the arm region*, which happens to
coincide with the label here. The **delta** subtracts that baseline (the real-goal clip is the
built-in "correct-dream control"): once the arm's tracking-hardness is removed, the dream's cycle
error is **indistinguishable** from the correct goal's (0.51). `viz/s4_on_dreams_examples.png`
shows the delta map (col 3) near-zero and unaligned with the cyan label, while invisibility (col 4)
is high across the whole arm regardless of the label.

### Why this matters (the contrast with S2/S4)
- **Synthetic S2 corruptions** (cut-paste / tps / composite) create a **local geometric
  inconsistency** — a pasted patch corresponds to nothing in the other frames → high cycle error
  *there*. S4 delta caught them (**0.78 AUROC**).
- **Real generative dreams** are **globally coherent but semantically wrong** — a wrong-but-plausible
  arm pose still self-corresponds fine → **no** local inconsistency → S4 delta **fails (0.51)**.
- **⇒ The S2 synthetic corruptions are not representative of real dreamer failures.** A disagreement
  detector trained on backbone geometric self-consistency (absolute *or* relative-to-synthetic-
  corruptions) will not transfer to real dreams. This **confirms and extends the M1 premise finding**
  on a real dreamer: the disagreement signal cannot come from the frozen backbone's own cycle /
  visibility self-consistency. It needs **cross-consistency** (does the dream agree with what
  *multiple observed frames* imply?) or a learned semantic critic over dream+obs features.

A *stronger* dreamer would only sharpen this: cleaner, more coherent dreams → even less local
geometric inconsistency → delta even closer to chance.

## Reproduce (env `gd4d5090`, GPU0)

```bash
cd /workspace/code/GD_4D
export HF_HOME=/workspace/huggingface_cache/
PY=/workspace/miniconda3/envs/gd4d5090/bin/python
BB=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG

# 1. train the dreamer (~46 min, GPU0) -> ckpt/unet_final
CUDA_VISIBLE_DEVICES=0 $PY experiments/M2_libero_dreamer/train.py --steps 10000 --batch 16 \
    --out experiments/M2_libero_dreamer/ckpt

# 2. generate dreams on held-out demos -> dreams/dreams.npz + dreams/item*.png
CUDA_VISIBLE_DEVICES=0 $PY experiments/M2_libero_dreamer/generate.py \
    --unet experiments/M2_libero_dreamer/ckpt/unet_final --out experiments/M2_libero_dreamer/dreams

# 3. characterize failure modes -> viz/dream_failure_gallery.png + viz/characterize_summary.json
$PY experiments/M2_libero_dreamer/characterize.py \
    --dreams experiments/M2_libero_dreamer/dreams/dreams.npz --out experiments/M2_libero_dreamer/viz

# 4. S4 geometric signal on real dreams -> viz/s4_on_dreams_report.json + viz/s4_on_dreams_examples.png
CUDA_VISIBLE_DEVICES=0 $PY experiments/M2_libero_dreamer/s4_on_dreams.py \
    --config $BB/model.yaml --ckpt $BB/opend4rt.ckpt \
    --dreams experiments/M2_libero_dreamer/dreams/dreams.npz --out experiments/M2_libero_dreamer/viz
```

## Visual outputs (per project rule)
| file | produced by |
|---|---|
| `dreams/item{00..19}_*.png` — obs \| real goal \| dream strips | `generate.py` |
| `viz/dream_failure_gallery.png` — obs \| goal \| dream \| \|dream−goal\| heatmap, 20 rows | `characterize.py` |
| `viz/characterize_summary.json` — localization stats | `characterize.py` |
| `viz/s4_on_dreams_examples.png` — dream \| goal \| cycle-delta \| invisibility, cyan=label | `s4_on_dreams.py` |
| `viz/s4_on_dreams_report.json` — pooled + per-item AUROCs | `s4_on_dreams.py` |

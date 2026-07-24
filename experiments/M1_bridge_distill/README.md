# M1 · S3 bridge distillation — training loop (first increment)

Trains the LoRA bridge so the **student** (observed frames + dream) reproduces the **teacher** (full real
clip) query outputs. GPU0 only, teacher targets cached.

## Result — training machinery validated

Overfit 8 LIBERO tuples, 60 steps: distill loss **0.045 → 0.008 (82% down)**; gradients flow to the LoRA
adapters; the student converges toward the teacher. Figure: `viz/distill_loss.png`.

*Not yet active (next increments):* imagined-time wiring into the encoder forward + dream-token gating.
The residual ~0.008 loss is what those should reduce — this increment proves the loop, cache, loss, and
backprop, not final quality.

## Design

| | clip | query |
|---|---|---|
| **teacher** (frozen, cached) | `[o_{t-7..t}, f_{t+1..t+20}]` — 28 frames | o_t (frame 7) → goal (frame 27), ref0 coords |
| **student** (bridge, trained) | `[obs×8, dream, dream]` — 10 frames | o_t (frame 7) → dream (frame 8) |

Dream is **duplicated** so it forms the clean last temporal patch (the encoder patches 2 frames → 1, so
an odd-length clip would drop the dream). Loss = `SmoothL1(xyz) + 0.1·MSE(confidence)`.

## Reproduce (env `gd4d5090`, GPU0)

```bash
cd /workspace/code/GD_4D
CKPT=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/gd4d5090/bin/python src/train/distill.py \
  --config $CKPT/model.yaml --ckpt $CKPT/opend4rt.ckpt --n-tuples 8 --steps 60 --batch 4
# -> experiments/M1_bridge_distill/viz/distill_loss.png
```

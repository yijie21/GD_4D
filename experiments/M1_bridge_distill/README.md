# M1 · S3 bridge distillation — training loop (first increment)

Trains the LoRA bridge so the **student** (observed frames + dream) reproduces the **teacher** (full real
clip) query outputs. GPU0 only, teacher targets cached.

## Status: full training path complete; **acceptance gate NOT yet passed** (needs iteration)

**Machinery — ✅ complete and working.** Bidirectional queries (o_t↔dream), dreams at all offsets
j∈{5,10,15,20}, imagined-time + dream-token gating active (`wrapper.py::encode_video`; identity still
`0.00e0`), **gradient checkpointing** (fits the 1.16 B backbone on one GPU), teacher targets cached,
held-out split. `distill_loss.png` (overfit-8 sanity, 85% down) + `distill_recall.png` (full run).

**First real run** (40 train / 12 held-out tuples, 150 steps, GPU0):

| | held-out recall@3px (fwd / bwd) |
|---|---|
| before (untrained) | 0.910 / 0.890 |
| after 150 steps | 0.909 / 0.896 |
| **gate ⛔** | **≥ 0.95** |

Train loss dropped 0.16 → 0.027, but **held-out recall did not improve** → this run does **not** pass the gate.

**Honest read (not a bug — grads flow, loss drops):**
1. The **untrained student is already ~0.91** — the frozen backbone natively handles `obs+dream`, so the
   bridge's headroom in this metric is small (consistent with the S4 de-risk).
2. **Overfitting** on 40 tuples (train loss ↓, held-out flat) — needs far more data.
3. The residual ~9% are likely **hard points** (occlusion / out-of-frame) where uv-recall@3px may be an
   unrealistic bar; the metric/threshold (and maybe visible-only scoring) needs reconsidering.

**Open — needs a decision:** scale data+steps (test overfitting), refine the metric/gate, or revisit
whether the bridge is even needed here. See the plan §M1.

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

# checkpoints/

Downloaded model weights for GD-4D. **All weights go here; never committed to git.**
Record every download below (map it to the component # in the plan's backbone inventory).

## Log

| Component (plan §1) | File | Source URL | sha256 | License | Date |
|---------------------|------|-----------|--------|---------|------|
| #1 D4RT 4D backbone (OpenD4RT, 48CLIP SOTA) | `OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt` (13 GB) | https://huggingface.co/Lijiaxin0111/OpenD4RT (`checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/`) | `65936f72eb902f9a3629e25b2888e8cc2ebf87a89bee7103fe94313fb53dd179` | Apache-2.0 | 2026-07-24 |

Download (into this dir, per the layout rule):
```bash
/workspace/miniconda3/envs/gd4d5090/bin/hf download Lijiaxin0111/OpenD4RT \
  checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml \
  --repo-type model --local-dir /workspace/code/GD_4D
```
Verified working: loads into the vendored `D4RTModel` and recovers correspondences — see
`experiments/S0_backbone_sanity/`. (The `48CLIP_9Mix_NoCropAUG` variant was chosen over
`32CLIP_9Dataset_NoAUG` as the SOTA/longer-window checkpoint.)

<!-- Example row:
| #2 DINOv2 ViT-L/14 | dinov2_vitl14.pth | https://... | <sha256> | Apache-2.0 | 2026-07-24 |
-->

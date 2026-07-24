# checkpoints/

Downloaded model weights for GD-4D. **All weights go here; never committed to git.**
Record every download below (map it to the component # in the plan's backbone inventory).

## Log

| Component (plan §1) | File | Source URL | sha256 | License | Date |
|---------------------|------|-----------|--------|---------|------|
| #1 D4RT 4D backbone (OpenD4RT, 48CLIP SOTA) | `OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt` (13 GB) | https://huggingface.co/Lijiaxin0111/OpenD4RT (`checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/`) | `65936f72eb902f9a3629e25b2888e8cc2ebf87a89bee7103fe94313fb53dd179` | Apache-2.0 | 2026-07-24 |
| #6 SAM 2 (masks for S2 corruptions) | `sam2/sam2.1_hiera_large.pt` (857 MB) | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | Apache-2.0 | 2026-07-24 |

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

SAM 2 download + verified working (23 clean per-object masks on a LIBERO goal frame):
```bash
curl -fL -o checkpoints/sam2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# loader: build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2/sam2.1_hiera_large.pt")
```

<!-- Example row:
| #2 DINOv2 ViT-L/14 | dinov2_vitl14.pth | https://... | <sha256> | Apache-2.0 | 2026-07-24 |
-->

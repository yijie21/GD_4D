"""Generate goal dreams from the fine-tuned LIBERO dreamer on held-out demos.

Loads the InstructPix2Pix pipeline, swaps in our fine-tuned UNet, and for each
held-out (obs, real_goal, instruction) item generates a dreamed goal image. Saves
a .npz of (obs, real_goal, dream) uint8 triples plus a per-item PNG strip.

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    export HF_HOME=/workspace/huggingface_cache/
    CUDA_VISIBLE_DEVICES=0 python experiments/M2_libero_dreamer/generate.py \
        --unet experiments/M2_libero_dreamer/ckpt/unet_final \
        --out experiments/M2_libero_dreamer/dreams
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from data import load_libero_spatial, val_eval_items  # noqa: E402

MODEL_ID = "timbrooks/instruct-pix2pix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", type=str, default=str(_HERE / "ckpt/unet_final"))
    ap.add_argument("--out", type=str, default=str(_HERE / "dreams"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--offset", type=int, default=25)
    ap.add_argument("--n_per_task", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--image_guidance", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

    print("[load] pipeline + fine-tuned unet:", args.unet)
    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float16)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    tasks = load_libero_spatial()
    items = val_eval_items(tasks, n_per_task=args.n_per_task, offset=args.offset, size=args.size)
    print(f"[gen] {len(items)} held-out items  (guidance={args.guidance} "
          f"image_guidance={args.image_guidance} steps={args.steps})")

    obs_all, goal_all, dream_all, meta = [], [], [], []
    for i, it in enumerate(items):
        obs_pil = Image.fromarray(it["obs_u8"]).resize((args.size, args.size), Image.BICUBIC)
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        dream = pipe(
            prompt=it["instruction"], image=obs_pil,
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            image_guidance_scale=args.image_guidance, generator=gen,
        ).images[0]
        obs_u8 = np.asarray(obs_pil, dtype=np.uint8)
        goal_u8 = np.asarray(Image.fromarray(it["goal_u8"]).resize((args.size, args.size), Image.BICUBIC), np.uint8)
        dream_u8 = np.asarray(dream.resize((args.size, args.size)), dtype=np.uint8)
        obs_all.append(obs_u8); goal_all.append(goal_u8); dream_all.append(dream_u8)
        meta.append({"task": it["task"], "instruction": it["instruction"],
                     "demo": it["demo"], "t0": it["t0"], "g": it["g"]})
        # per-item strip: obs | real goal | dream
        strip = np.concatenate([obs_u8, goal_u8, dream_u8], axis=1)
        Image.fromarray(strip).save(out / f"item{i:02d}_{it['task'][:30]}.png")
        print(f"  [{i:2d}] {it['task'][:44]:44s} done")

    np.savez_compressed(
        out / "dreams.npz",
        obs=np.stack(obs_all), goal=np.stack(goal_all), dream=np.stack(dream_all),
        instruction=np.array([m["instruction"] for m in meta]),
        task=np.array([m["task"] for m in meta]),
        demo=np.array([m["demo"] for m in meta]),
        t0=np.array([m["t0"] for m in meta]),
        g=np.array([m["g"] for m in meta]),
    )
    print(f"[save] {out/'dreams.npz'}  ({len(obs_all)} triples) + PNG strips")


if __name__ == "__main__":
    main()

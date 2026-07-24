"""Generate correct vs wrong-instruction dreams from the 4-suite dreamer.

Closes the synthetic-corruption gap: instead of S2 edits, we get REAL generative
dream errors. For each held-out obs we generate:
  * dream_correct : dreamer(o_t, the scene's OWN instruction)          -> right goal
  * dream_wrong   : dreamer(o_t, a DIFFERENT task's instruction)        -> wrong goal, same scene
Both are generated (no real-vs-fake shortcut) and share the scene; they differ only
in whether the goal is correct for this observation. With the 4 diverse suites, a
wrong instruction ("open the drawer" on a soup-in-basket scene) yields a genuinely
different goal.

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    export HF_HOME=/workspace/huggingface_cache/
    CUDA_VISIBLE_DEVICES=0 python experiments/M2_libero_dreamer/gen_correct_wrong.py \
        --unet experiments/M2_libero_dreamer/ckpt_4suite/unet_final
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
from data import load_libero  # noqa: E402

MODEL_ID = "timbrooks/instruct-pix2pix"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", type=str, default=str(_HERE / "ckpt_4suite/unet_final"))
    ap.add_argument("--out", type=str, default=str(_HERE / "dreams_cw"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--offset", type=int, default=30)
    ap.add_argument("--per_task", type=int, default=2)
    ap.add_argument("--val_demos", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    # stronger text guidance + lower image guidance -> the (weakly-conditioned) IP2P
    # dreamer actually follows the instruction, so a wrong instruction yields a
    # genuinely different goal (|correct-wrong| ~28px vs ~6px at 7.5/1.5).
    ap.add_argument("--guidance", type=float, default=12.0)
    ap.add_argument("--image_guidance", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float16)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID, unet=unet, safety_checker=None, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)

    tasks = load_libero(SUITES, max_demos=50)
    all_instr = [(t["suite"], t["instruction"]) for t in tasks]
    rng = np.random.default_rng(1)

    def to256(u8):
        return np.asarray(Image.fromarray(u8).resize((args.size, args.size), Image.BICUBIC), np.uint8)

    obs_a, corr_a, wrong_a, goal_a, meta = [], [], [], [], []
    k = 0
    for ti, t in enumerate(tasks):
        n = len(t["demos"])
        val_ids = list(range(n - args.val_demos, n))[:args.per_task]
        for di in val_ids:
            fr = t["demos"][di]
            T = fr.shape[0]
            t0 = T // 3
            g = min(t0 + args.offset, T - 1)
            obs_u8 = to256(fr[t0])
            obs_pil = Image.fromarray(obs_u8)
            # wrong instruction: a task from a DIFFERENT suite
            others = [ins for (su, ins) in all_instr if su != t["suite"]]
            wrong_instr = others[int(rng.integers(len(others)))]

            def gen(instr, seed):
                ggen = torch.Generator(device=device).manual_seed(seed)
                return np.asarray(pipe(prompt=instr, image=obs_pil, num_inference_steps=args.steps,
                                       guidance_scale=args.guidance, image_guidance_scale=args.image_guidance,
                                       generator=ggen).images[0].resize((args.size, args.size)), np.uint8)
            dc = gen(t["instruction"], 100 + k)
            dw = gen(wrong_instr, 200 + k)
            obs_a.append(obs_u8); corr_a.append(dc); wrong_a.append(dw); goal_a.append(to256(fr[g]))
            meta.append({"suite": t["suite"], "task": t["task"], "demo": di,
                         "instr": t["instruction"], "wrong_instr": wrong_instr})
            if k < 24:
                strip = np.concatenate([obs_u8, to256(fr[g]), dc, dw], axis=1)
                Image.fromarray(strip).save(out / f"cw{k:02d}_{t['suite']}.png")
            k += 1
        if (ti + 1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} tasks, {k} items", flush=True)

    np.savez_compressed(out / "dreams_cw.npz",
                        obs=np.stack(obs_a), dream_correct=np.stack(corr_a),
                        dream_wrong=np.stack(wrong_a), goal=np.stack(goal_a),
                        suite=np.array([m["suite"] for m in meta]),
                        instr=np.array([m["instr"] for m in meta]),
                        wrong_instr=np.array([m["wrong_instr"] for m in meta]))
    print(f"[save] {out/'dreams_cw.npz'}  ({k} items) + {min(k,24)} strips (obs|goal|correct|wrong)")


if __name__ == "__main__":
    main()

"""Fine-tune a SuSIE-style goal-image dreamer on LIBERO (InstructPix2Pix recipe).

SuSIE = InstructPix2Pix fine-tuned on (current-obs, instruction, future-frame)
robot triples. We run that exact recipe on LIBERO so the dreamer lives in the
*deployment* distribution: init from `timbrooks/instruct-pix2pix` (already an
8-channel image-conditioned SD1.5 UNet), freeze VAE + text encoder, train the
UNet on libero_spatial goal pairs.

Single GPU (GPU0), bf16 autocast, gradient checkpointing. No accelerate launcher.

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    export HF_HOME=/workspace/huggingface_cache/
    CUDA_VISIBLE_DEVICES=0 python experiments/M2_libero_dreamer/train.py \
        --steps 8000 --batch 16 --out experiments/M2_libero_dreamer/ckpt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from data import LiberoGoalPairs, load_libero_spatial  # noqa: E402

MODEL_ID = "timbrooks/instruct-pix2pix"
VAE_SCALE = 0.18215


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--offset_min", type=int, default=15)
    ap.add_argument("--offset_max", type=int, default=30)
    ap.add_argument("--cond_drop", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out", type=str, default=str(_HERE / "ckpt"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    print("[load] components from", MODEL_ID)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device)
    noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    print(f"[load] unet in_channels={unet.config.in_channels} (expect 8)")

    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    text_encoder.to(torch.bfloat16)
    vae.to(torch.bfloat16)
    unet.train()
    unet.enable_gradient_checkpointing()

    # Precompute the empty-string ("dropped text") embedding once.
    with torch.no_grad():
        empty = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to(device)
        empty_emb = text_encoder(empty)[0].to(torch.bfloat16)  # [1,77,768]

    tasks = load_libero_spatial()
    ds = LiberoGoalPairs(tasks, split="train", offset_min=args.offset_min,
                         offset_max=args.offset_max, size=args.size,
                         length=args.steps * args.batch)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    @torch.no_grad()
    def encode_text(instructions):
        tok = tokenizer(list(instructions), padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        return text_encoder(tok)[0].to(torch.bfloat16)

    @torch.no_grad()
    def vae_encode(x, sample=True):
        d = vae.encode(x.to(torch.bfloat16)).latent_dist
        z = d.sample() if sample else d.mode()
        return z * VAE_SCALE

    print(f"[train] steps={args.steps} batch={args.batch} lr={args.lr} "
          f"(offsets {args.offset_min}-{args.offset_max})")
    step, t0, loss_acc = 0, time.time(), 0.0
    for batch in dl:
        obs = batch["obs"].to(device, non_blocking=True)
        goal = batch["goal"].to(device, non_blocking=True)
        B = obs.shape[0]

        z_goal = vae_encode(goal, sample=True)     # target latent (noised)
        z_obs = vae_encode(obs, sample=False)      # conditioning latent (clean)
        txt = encode_text(batch["instruction"])    # [B,77,768]

        # Classifier-free-guidance dropout (independent text / image drop).
        drop_txt = torch.rand(B, device=device) < args.cond_drop
        drop_img = torch.rand(B, device=device) < args.cond_drop
        if drop_txt.any():
            txt = torch.where(drop_txt[:, None, None], empty_emb.expand_as(txt), txt)
        if drop_img.any():
            z_obs = torch.where(drop_img[:, None, None, None], torch.zeros_like(z_obs), z_obs)

        noise = torch.randn_like(z_goal)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (B,), device=device).long()
        z_noisy = noise_sched.add_noise(z_goal, noise, t)
        unet_in = torch.cat([z_noisy, z_obs], dim=1)  # 8 channels

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(unet_in, t, encoder_hidden_states=txt).sample
        loss = F.mse_loss(pred.float(), noise.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()

        loss_acc += loss.item()
        step += 1
        if step % 50 == 0:
            dt = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss {loss_acc/50:.4f}  "
                  f"{50*args.batch/dt:.1f} img/s  ({dt:.0f}s/50)", flush=True)
            loss_acc, t0 = 0.0, time.time()
        if step % args.save_every == 0 or step == args.steps:
            ck = out / (f"unet_step{step}" if step < args.steps else "unet_final")
            unet.save_pretrained(ck)
            print(f"  [save] {ck}", flush=True)
        if step >= args.steps:
            break

    print("[done] final unet at", out / "unet_final")


if __name__ == "__main__":
    main()

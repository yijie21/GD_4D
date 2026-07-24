"""Premise test — is a GENERATED dream out-of-distribution for the frozen backbone?

The bridge (M1) only earns its keep if native-encode FAILS on generated dreams (unlike real frames,
which it handles at ~0.91). We native-encode `[obs, dream, dream]` and measure the forward→backward
**cycle error** (o_t→dream→o_t, px) + **visibility** of a grid, for three dreams:
  * real     — the true reached goal f_{t+n}          (in-distribution baseline)
  * halluc   — InstructPix2Pix @ image_guidance 1.5   (aggressive; reimagines the scene)
  * preserved— InstructPix2Pix @ image_guidance 2.75  (conservative; ~copies o_t)

High cycle error / low visibility on a generated dream ⇒ it is OOD ⇒ the bridge has a real job.
(Also probes whether IP2P is even a usable dreamer here.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SRC)); sys.path.insert(0, str(_SRC / "eval"))

import s0_backbone_sanity as s0  # noqa: E402
from data import WindowConfig, build_tuple, plan_tuples  # noqa: E402
from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402

OUT = Path(__file__).resolve().parent
CKPT = "checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"
N_TUPLES = 5


def r256(f):
    return cv2.resize(np.ascontiguousarray(f), (256, 256), interpolation=cv2.INTER_LINEAR)


def gather(n):
    cfg = WindowConfig(); out = []
    for f in sorted((LIBERO_ROOT / "libero_spatial").glob("*.hdf5")):
        tr = LiberoTrajectory(f, 0)
        sp = plan_tuples(tr.episode_id, len(tr), cfg)
        if sp:
            out.append(build_tuple(tr, sp[len(sp) // 2]))
        if len(out) >= n:
            break
    return out


def gen_dreams(tuples):
    from PIL import Image
    from diffusers import StableDiffusionInstructPix2PixPipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to("cuda")
    dreams = []
    for tup in tuples:
        o_t = Image.fromarray(cv2.resize(tup.obs_frames[-1], (512, 512)))
        d = {"real": r256(tup.goal_frame)}
        for name, igs in (("halluc", 1.5), ("preserved", 2.75)):
            g = torch.Generator("cuda").manual_seed(0)
            im = pipe(tup.instruction, image=o_t, num_inference_steps=20,
                      image_guidance_scale=igs, guidance_scale=5.0, generator=g).images[0]
            d[name] = np.array(im.resize((256, 256)))
        dreams.append(d)
    del pipe; torch.cuda.empty_cache()
    return dreams


def cycle_and_vis(model, obs_frames, dream_256):
    device = next(model.parameters()).device
    clip = np.stack([r256(f) for f in obs_frames] + [dream_256, dream_256])          # [10,256,256,3]
    vb = torch.from_numpy(clip).to(device, torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    asp = torch.tensor([[1.0]], device=device)
    xs = np.linspace(0.08, 0.92, 24, dtype=np.float32)
    uv = np.stack(np.meshgrid(xs, xs, indexing="xy"), -1).reshape(-1, 2)
    n = uv.shape[0]; s7 = np.full((n,), 7, np.int64); s8 = np.full((n,), 8, np.int64)
    with torch.no_grad():
        mem = s0._encode_model_memory(model=model, video_b=vb, aspect_b=asp)
        fwd = s0._query(model, vb, asp, mem, uv, s7, s8, s8, 4096)                    # o_t → dream
        bwd = s0._query(model, vb, asp, mem, fwd["uv_2d"], s8, s7, s7, 4096)          # dream → o_t
    cyc = np.linalg.norm((bwd["uv_2d"] - uv) * 255.0, axis=1)
    vis = 1.0 / (1.0 + np.exp(-fwd["visibility"]))
    return float(np.median(cyc)), float(vis.mean())


def main():
    tuples = gather(N_TUPLES)
    print(f"tuples={len(tuples)}; generating dreams (InstructPix2Pix)…")
    dreams = gen_dreams(tuples)

    model = s0.load_backbone(f"{CKPT}/model.yaml", f"{CKPT}/opend4rt.ckpt", s0._resolve_device("cuda:0"))
    conds = ["real", "halluc", "preserved"]
    res = {k: {"cyc": [], "vis": []} for k in conds}
    for tup, d in zip(tuples, dreams):
        for k in conds:
            c, v = cycle_and_vis(model, tup.obs_frames, d[k])
            res[k]["cyc"].append(c); res[k]["vis"].append(v)
    print("\ncondition   median cycle-err (px)   mean visibility")
    for k in conds:
        print(f"  {k:10s}  {np.mean(res[k]['cyc']):6.2f}                {np.mean(res[k]['vis']):.3f}")

    # example strip: o_t | real | halluc | preserved
    (OUT / "viz").mkdir(parents=True, exist_ok=True)
    ex = tuples[0]; d0 = dreams[0]
    strip = np.concatenate([r256(ex.obs_frames[-1]), d0["real"], d0["halluc"], d0["preserved"]], 1)[:, :, ::-1]
    cv2.imwrite(str(OUT / "viz" / "dream_examples.png"), strip)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    col = ["#54a24b", "#e45756", "#4c78a8"]
    a1.bar(conds, [np.mean(res[k]["cyc"]) for k in conds], color=col)
    a1.set_ylabel("median cycle error (px)"); a1.set_title("correspondence consistency (lower=better)")
    a2.bar(conds, [np.mean(res[k]["vis"]) for k in conds], color=col)
    a2.set_ylabel("mean visibility"); a2.set_title("visibility (higher=better)"); a2.set_ylim(0, 1)
    fig.suptitle("Premise: is a generated dream OOD for the frozen backbone?", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "viz" / "premise.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "viz" / "premise.png")


if __name__ == "__main__":
    main()

"""S0 visuals — *show* the recovered correspondences (companion to s0_backbone_sanity.py).

Produces, per clip:
  1. corr_lines_<tag>_t{b}.png  — source frame | target frame, side by side, with lines
                                   connecting each query pixel to its predicted match at frame b.
                                   Point colour = a smooth 2-D field over the source grid, so a
                                   viewer can see the spatial arrangement is preserved under motion.
  2. cycle_<tag>_t{b}.png        — one source frame: original points (o) vs the forward->backward
                                   round-trip points (x) with a segment between; short segment = good.
  3. trails_<tag>.png            — forward tracks 0->t drawn as poly-lines (colour = time), montage.
  4. track_<tag>.gif             — the grid tracked across all frames (occluded points dimmed).

Run: see experiments/S0_backbone_sanity/README.md.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# reuse the S0 backbone loaders / query helpers (same directory)
from s0_backbone_sanity import (  # noqa: E402
    IMAGE_HW, _query, _sigmoid, _uniform_indices,
    load_backbone, load_libero_frames, load_video_frames,
)
from src.eval.tasks import _encode_model_memory, _model_clip_frames  # noqa: E402 (vendored)
from infer_track_3d import _resolve_device  # noqa: E402 (vendored)


def point_colors(uv0: np.ndarray) -> np.ndarray:
    """Smooth 2-D colour field over the source grid: hue<-x, value<-y. Returns rgb [N,3] in [0,1]."""
    out = np.zeros((uv0.shape[0], 3), dtype=np.float32)
    for i, (x, y) in enumerate(uv0):
        out[i] = colorsys.hsv_to_rgb(0.85 * float(x), 0.85, 0.45 + 0.5 * float(y))
    return out


def forward_tracks(model, frames, grid, chunk):
    """Forward-track a source grid (frame 0) across all frames. Returns uv0[N,2], tracks[N,T,2], vis[N,T]."""
    device = next(model.parameters()).device
    T, H, W, _ = frames.shape
    video_b = (torch.from_numpy(frames).to(device=device, dtype=torch.float32)
               .permute(0, 3, 1, 2).unsqueeze(0) / 255.0)
    aspect_b = torch.tensor([[float(W) / float(H)]], dtype=torch.float32, device=device)
    with torch.no_grad():
        memory = _encode_model_memory(model=model, video_b=video_b, aspect_b=aspect_b)
        m = 0.08
        xs = np.linspace(m, 1 - m, grid, dtype=np.float32)
        ys = np.linspace(m, 1 - m, grid, dtype=np.float32)
        uv0 = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        N = uv0.shape[0]
        allt = np.tile(np.arange(T, dtype=np.int64), N)
        alluv = np.repeat(uv0, T, axis=0)
        src = np.zeros((N * T,), dtype=np.int64)
        pred = _query(model, video_b, aspect_b, memory, alluv, src, allt, allt, chunk)
        tracks = pred["uv_2d"].reshape(N, T, 2)
        vis = _sigmoid(pred["visibility"].reshape(N, T)) > 0.5
        # backward round-trip for the cycle figure: for each target b, (uv_b, b->0)
        cyc = {}
        for b in _cycle_targets(T):
            uvb = tracks[:, b]
            tb = np.full((N,), b, dtype=np.int64)
            back = _query(model, video_b, aspect_b, memory, uvb, tb,
                          np.zeros((N,), np.int64), np.zeros((N,), np.int64), chunk)["uv_2d"]
            cyc[b] = back
        return uv0, tracks, vis, cyc, memory


def _cycle_targets(T):
    return sorted(set(int(round(x)) for x in [T * 0.5, T - 1] if 0 < x <= T - 1))


def _scale():
    H, W = IMAGE_HW
    return np.array([W - 1, H - 1], dtype=np.float32)


def fig_corr_lines(frames, uv0, tracks, vis, colors, src_idx, tgt_idx, out):
    """Side-by-side source|target with connecting match lines (visible matches only)."""
    H, W = IMAGE_HW
    sc = _scale()
    canvas = np.concatenate([frames[src_idx], frames[tgt_idx]], axis=1)  # [H, 2W, 3] RGB
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=130)
    ax.imshow(canvas)
    ax.axis("off")
    p_src = uv0 * sc                                   # [N,2] px in left image
    p_tgt = tracks[:, tgt_idx] * sc                    # [N,2] px in right image
    p_tgt = p_tgt + np.array([W, 0.0])                 # shift into right half
    v = vis[:, tgt_idx]
    for i in range(uv0.shape[0]):
        if not v[i]:
            continue
        ax.plot([p_src[i, 0], p_tgt[i, 0]], [p_src[i, 1], p_tgt[i, 1]],
                "-", color=colors[i], linewidth=0.7, alpha=0.7)
    ax.scatter(p_src[v, 0], p_src[v, 1], s=14, c=colors[v], edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(p_tgt[v, 0], p_tgt[v, 1], s=14, c=colors[v], edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_title(f"correspondence:  frame {src_idx}  →  frame {tgt_idx}      "
                 f"{int(v.sum())}/{len(v)} visible matches drawn", fontsize=11, pad=10)
    bb = dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55)
    ax.text(W * 0.5, H - 6, f"source (t={src_idx})", ha="center", va="bottom",
            color="white", fontsize=10, bbox=bb)
    ax.text(W * 1.5, H - 6, f"target (t={tgt_idx})", ha="center", va="bottom",
            color="white", fontsize=10, bbox=bb)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cycle(frames, uv0, back_uv, vis_b, colors, src_idx, tgt_idx, out):
    """Forward->backward round-trip on the source frame, VISIBLE points only (matches the metric):
    original (o) vs returned (x). Occluded/out-of-frame points have no meaningful round-trip, so
    they are excluded — exactly as the S0 cycle metric is computed."""
    sc = _scale()
    fig, ax = plt.subplots(figsize=(6.4, 6.6), dpi=130)
    ax.imshow(frames[src_idx]); ax.axis("off")
    p0 = (uv0 * sc)[vis_b]
    pb = (back_uv * sc)[vis_b]
    cols = colors[vis_b]
    err = np.linalg.norm(p0 - pb, axis=1) if len(p0) else np.array([0.0])
    for i in range(p0.shape[0]):
        ax.plot([p0[i, 0], pb[i, 0]], [p0[i, 1], pb[i, 1]], "-", color=cols[i], linewidth=0.9, alpha=0.9)
    ax.scatter(p0[:, 0], p0[:, 1], s=26, facecolors="none", edgecolors=cols, linewidths=1.3, zorder=3)
    ax.scatter(pb[:, 0], pb[:, 1], s=18, marker="x", c=cols, linewidths=1.3, zorder=3)
    ax.set_title(f"fwd–bwd cycle  0 → {tgt_idx} → 0   (visible pts only)\n"
                 f"median {np.median(err):.1f}px   ·   {int(vis_b.sum())} pts   ·   "
                 f"o = start, x = returned", fontsize=11, pad=8)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_trails(frames, tracks, vis, out, n_frames=6):
    """Montage: per-frame image with each point's path 0->t drawn (colour = time)."""
    sc = _scale()
    T = frames.shape[0]
    N = tracks.shape[0]
    pick = _uniform_indices(T, n_frames)
    fig, axes = plt.subplots(1, len(pick), figsize=(3.0 * len(pick), 3.2), dpi=120)
    cmap = plt.cm.plasma
    for ax, t in zip(np.atleast_1d(axes), pick):
        ax.imshow(frames[t]); ax.axis("off"); ax.set_title(f"t={int(t)}", fontsize=10)
        for i in range(0, N, 2):                                   # subsample for clarity
            path = tracks[i, : t + 1] * sc
            vv = vis[i, : t + 1]
            if vv.sum() < 2:
                continue
            for k in range(1, t + 1):
                if vis[i, k] and vis[i, k - 1]:
                    ax.plot(path[k - 1:k + 1, 0], path[k - 1:k + 1, 1],
                            "-", color=cmap(k / max(T - 1, 1)), linewidth=0.6, alpha=0.8)
    fig.suptitle("forward tracks  (colour = time, brighter = later)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def make_gif(frames, tracks, vis, colors, out, upscale=2, fps=8, tail=6):
    """Animate the tracked grid; occluded points dimmed; short motion tail."""
    sc = _scale()
    T, H, W, _ = frames.shape
    bgr_colors = (colors[:, ::-1] * 255).astype(np.uint8)
    gif = []
    for t in range(T):
        img = np.ascontiguousarray(frames[t][:, :, ::-1])         # RGB->BGR
        img = cv2.resize(img, (W * upscale, H * upscale), interpolation=cv2.INTER_NEAREST)
        for i in range(tracks.shape[0]):
            for k in range(max(1, t - tail + 1), t + 1):
                if vis[i, k] and vis[i, k - 1]:
                    p0 = (tracks[i, k - 1] * sc * upscale).astype(int)
                    p1 = (tracks[i, k] * sc * upscale).astype(int)
                    cv2.line(img, tuple(p0), tuple(p1), tuple(int(c) for c in bgr_colors[i]), 1, cv2.LINE_AA)
            p = (tracks[i, t] * sc * upscale).astype(int)
            col = tuple(int(c) for c in bgr_colors[i])
            if vis[i, t]:
                cv2.circle(img, tuple(p), 3, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, tuple(p), 2, tuple(int(0.4 * c) for c in col), 1, cv2.LINE_AA)
        cv2.putText(img, f"t={t}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        gif.append(img[:, :, ::-1])                               # BGR->RGB
    imageio.mimsave(out, gif, fps=fps, loop=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize recovered correspondences (S0).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--libero-hdf5")
    ap.add_argument("--demo-index", type=int, default=0)
    ap.add_argument("--cam", default="agentview_rgb")
    ap.add_argument("--num-frames", type=int, default=48)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    model = load_backbone(args.config, args.ckpt, device)
    frames = (load_video_frames(args.video, args.num_frames) if args.video
              else load_libero_frames(args.libero_hdf5, args.demo_index, args.cam, args.num_frames))
    T = frames.shape[0]

    uv0, tracks, vis, cyc, _ = forward_tracks(model, frames, args.grid, args.chunk)
    colors = point_colors(uv0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    targets = _cycle_targets(T)
    for b in [_uniform_indices(T, 3)[1]] + targets:                # a mid + the cycle targets
        b = int(b)
        fig_corr_lines(frames, uv0, tracks, vis, colors, 0, b, out / f"corr_lines_{args.tag}_t{b}.png")
    for b in targets:
        fig_cycle(frames, uv0, cyc[b], vis[:, b], colors, 0, int(b), out / f"cycle_{args.tag}_t{b}.png")
    fig_trails(frames, tracks, vis, out / f"trails_{args.tag}.png")
    make_gif(frames, tracks, vis, colors, out / f"track_{args.tag}.gif")
    print("wrote:", *(p.name for p in sorted(out.glob(f"*{args.tag}*"))), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

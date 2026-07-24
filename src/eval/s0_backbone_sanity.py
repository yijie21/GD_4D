"""S0 · Backbone sanity — are correspondences recoverable with the released D4RT weights?

This is the de-risking gate before M1 (the bridge). The whole GD-4D idea rests on one
assumption: the frozen 4D backbone can, via its point-query interface X(u, t_src -> t_tgt),
recover *geometrically consistent* correspondences with *calibrated* confidence/visibility.
Both GD-4D heads read the four scalars this interface returns (xyz_3d, uv_2d, visibility,
confidence), so if recovery is noise, everything above it collapses.

We verify this WITHOUT ground-truth motion, using self-consistency probes that any working
tracker must pass:

  1. Identity     : query (u, t_src=a -> t_tgt=a) must return uv ~= (u, v).            [head/coord sanity]
  2. Fwd-bwd cycle: query (u, a->b) -> uv_b, then (uv_b, b->a) -> uv'; uv' ~= (u, v).   [the real signal; == our e_cyc feature]
  3. Calibration  : confidence / visibility(sigmoid) behave sensibly vs. horizon.       [validates c_bg, v features]
  4. Overlay      : forward-track a grid and render it, to eyeball that tracks follow the scene.

Coordinate conventions (verified against the vendored repo):
  - query u, v are NORMALIZED to [0, 1]  (u = px_x / (W-1), v = px_y / (H-1))
  - model uv_2d output is NORMALIZED [0, 1]  -> pixels via uv * (W-1, H-1)
  - visibility, confidence are LOGITS -> apply sigmoid

Run:  see experiments/S0_backbone_sanity/README.md  (fully reproducible).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# --- make the vendored OpenD4RT package importable (its top-level pkg is also called `src`) ---
_REPO_ROOT = Path(__file__).resolve().parents[2]          # /workspace/code/GD_4D
_D4RT = _REPO_ROOT / "third_party" / "Open-d4rt"
if str(_D4RT) not in sys.path:
    sys.path.insert(0, str(_D4RT))                        # front, so `import src.*` -> vendored

from src.core import load_checkpoint, load_yaml_config      # noqa: E402  (vendored)
from src.model import build_model                            # noqa: E402  (vendored)
from src.eval.tasks import (                                 # noqa: E402  (vendored)
    _encode_model_memory,
    _model_clip_frames,
    _run_model_for_queries,
)
from infer_track_3d import _resolve_device, _unwrap_state_dict  # noqa: E402  (vendored)

IMAGE_HW = (256, 256)   # model input (see model.yaml: input.image_size)


# ---------------------------------------------------------------------------
# Video loaders  ->  [T, H, W, 3] uint8
# ---------------------------------------------------------------------------
def _uniform_indices(total: int, k: int) -> np.ndarray:
    if total <= k:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=k, dtype=np.int64)


def load_video_frames(path: str, num_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    raw = np.stack(frames, axis=0)
    idx = _uniform_indices(raw.shape[0], num_frames)
    picked = raw[idx]
    h, w = IMAGE_HW
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in picked], axis=0)


def load_libero_frames(hdf5: str, demo_index: int, cam: str, num_frames: int) -> np.ndarray:
    import h5py

    with h5py.File(hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        key = demos[demo_index]
        rgb = np.asarray(f["data"][key]["obs"][cam][:])   # [T,128,128,3] uint8, stored vertically flipped
    rgb = rgb[:, ::-1, :, :]                               # correct LIBERO/robomimic vertical flip
    idx = _uniform_indices(rgb.shape[0], num_frames)
    picked = rgb[idx]
    h, w = IMAGE_HW
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in picked], axis=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_backbone(config: str, ckpt: str, device: torch.device) -> torch.nn.Module:
    cfg = load_yaml_config(config)
    model = build_model(cfg["model"]).eval().to(device)
    payload = load_checkpoint(ckpt, map_location="cpu")
    state = _unwrap_state_dict(payload)
    if not state:
        raise RuntimeError(f"no weights found in {ckpt}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(unexpected)
    print(f"[load] params in ckpt={len(state)}  loaded~={n_loaded}  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval()
    return model


def _query(model, video_b, aspect_b, memory, uv_norm, t_src, t_tgt, t_cam, chunk):
    """uv_norm: [N,2] float; t_*: [N] int. Returns dict of numpy [N,...]."""
    dev = video_b.device
    q = {
        "u": torch.as_tensor(uv_norm[:, 0], dtype=torch.float32, device=dev),
        "v": torch.as_tensor(uv_norm[:, 1], dtype=torch.float32, device=dev),
        "t_src": torch.as_tensor(t_src, dtype=torch.long, device=dev),
        "t_tgt": torch.as_tensor(t_tgt, dtype=torch.long, device=dev),
        "t_cam": torch.as_tensor(t_cam, dtype=torch.long, device=dev),
    }
    pred = _run_model_for_queries(model=model, video_b=video_b, aspect_b=aspect_b,
                                  query=q, chunk_size=chunk, memory_b=memory)
    return {k: v.numpy() for k, v in pred.items()}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Spike
# ---------------------------------------------------------------------------
def run(model, frames_uint8: np.ndarray, grid: int, chunk: int, out_dir: Path, tag: str) -> dict:
    device = next(model.parameters()).device
    T, H, W, _ = frames_uint8.shape
    clip = _model_clip_frames(model)
    print(f"[spike:{tag}] frames={T} (model clip_frames={clip}) size={H}x{W}")

    video_b = (torch.from_numpy(frames_uint8).to(device=device, dtype=torch.float32)
               .permute(0, 3, 1, 2).unsqueeze(0) / 255.0)           # [1,T,3,H,W]
    aspect_b = torch.tensor([[float(W) / float(H)]], dtype=torch.float32, device=device)

    with torch.no_grad():
        memory = _encode_model_memory(model=model, video_b=video_b, aspect_b=aspect_b)

        # query grid at t_src=0 (interior margin to avoid border artifacts)
        m = 0.08
        xs = np.linspace(m, 1.0 - m, grid, dtype=np.float32)
        ys = np.linspace(m, 1.0 - m, grid, dtype=np.float32)
        uv0 = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)   # [N,2] normalized
        N = uv0.shape[0]
        scale = np.array([W - 1, H - 1], dtype=np.float32)

        def err_px(a_norm, b_norm):
            return np.linalg.norm((a_norm - b_norm) * scale, axis=-1)               # [N] pixels

        report: dict = {"tag": tag, "num_frames": int(T), "num_queries": int(N),
                        "image_hw": list(IMAGE_HW), "grid": int(grid)}

        # (1) IDENTITY: t_src=0 -> t_tgt=0
        z = np.zeros((N,), dtype=np.int64)
        p_id = _query(model, video_b, aspect_b, memory, uv0, z, z, z, chunk)
        id_err = err_px(p_id["uv_2d"], uv0)
        report["identity"] = {
            "median_px": float(np.median(id_err)), "mean_px": float(np.mean(id_err)),
            "p90_px": float(np.percentile(id_err, 90)),
            "mean_confidence": float(np.mean(_sigmoid(p_id["confidence"]))),
            "mean_visible_frac": float(np.mean(_sigmoid(p_id["visibility"]) > 0.5)),
        }

        # (2) FWD-BWD CYCLE for a set of target horizons
        targets = sorted(set(int(round(x)) for x in [T * 0.25, T * 0.5, T * 0.75, T - 1] if 0 < x <= T - 1))
        report["cycle"] = {}
        for b in targets:
            tb = np.full((N,), b, dtype=np.int64)
            fwd = _query(model, video_b, aspect_b, memory, uv0, z, tb, tb, chunk)      # 0 -> b (cam b)
            uv_b = fwd["uv_2d"]
            vis_b = _sigmoid(fwd["visibility"]) > 0.5
            bwd = _query(model, video_b, aspect_b, memory, uv_b, tb, z, z, chunk)       # b -> 0 (cam 0)
            cyc = err_px(bwd["uv_2d"], uv0)
            vis = cyc[vis_b] if vis_b.any() else cyc
            report["cycle"][f"h={b}"] = {
                "horizon_frac": round(b / (T - 1), 3),
                "median_px_visible": float(np.median(vis)),
                "median_px_all": float(np.median(cyc)),
                "p90_px_visible": float(np.percentile(vis, 90)),
                "visible_frac": float(np.mean(vis_b)),
                "mean_confidence_fwd": float(np.mean(_sigmoid(fwd["confidence"]))),
            }

        # (3) CALIBRATION vs horizon: forward-track grid across ALL frames, report per-frame stats
        allt = np.tile(np.arange(T, dtype=np.int64), N)
        alluv = np.repeat(uv0, T, axis=0)
        allsrc = np.zeros((N * T,), dtype=np.int64)
        track = _query(model, video_b, aspect_b, memory, alluv, allsrc, allt, allt, chunk)
        uv_tracks = track["uv_2d"].reshape(N, T, 2)          # normalized
        conf = _sigmoid(track["confidence"].reshape(N, T))
        visf = _sigmoid(track["visibility"].reshape(N, T)) > 0.5
        report["calibration"] = {
            "mean_confidence_by_frame": [float(x) for x in conf.mean(0)],
            "visible_frac_by_frame": [float(x) for x in visf.mean(0)],
        }

    # (4) OVERLAY: draw forward tracks on frames (subsample points), save a montage
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_overlay(frames_uint8, uv_tracks, visf, scale, out_dir / f"overlay_{tag}.png")
    report["overlay_png"] = str(out_dir / f"overlay_{tag}.png")

    (out_dir / f"report_{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def _save_overlay(frames_uint8, uv_tracks, visf, scale, path: Path) -> None:
    """Montage of ~6 frames with tracked points overlaid (green=visible, red=occluded)."""
    T, H, W, _ = frames_uint8.shape
    N = uv_tracks.shape[0]
    pick_pts = np.linspace(0, N - 1, min(N, 80), dtype=np.int64)
    pick_frames = _uniform_indices(T, min(T, 6))
    tiles = []
    for t in pick_frames:
        img = np.ascontiguousarray(frames_uint8[t][:, :, ::-1])   # RGB->BGR for cv2
        for pi in pick_pts:
            x, y = (uv_tracks[pi, t] * scale).astype(int)
            if 0 <= x < W and 0 <= y < H:
                color = (0, 200, 0) if visf[pi, t] else (0, 0, 220)
                cv2.circle(img, (int(x), int(y)), 2, color, -1)
        cv2.putText(img, f"t={int(t)}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    montage = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(path), montage)


def main() -> int:
    ap = argparse.ArgumentParser(description="S0 backbone-sanity: correspondence recoverability check.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to an mp4/avi/mov")
    src.add_argument("--libero-hdf5", help="path to a LIBERO .hdf5")
    ap.add_argument("--demo-index", type=int, default=0)
    ap.add_argument("--cam", default="agentview_rgb")
    ap.add_argument("--num-frames", type=int, default=48)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    model = load_backbone(args.config, args.ckpt, device)

    if args.video:
        frames = load_video_frames(args.video, args.num_frames)
    else:
        frames = load_libero_frames(args.libero_hdf5, args.demo_index, args.cam, args.num_frames)

    run(model, frames, grid=args.grid, chunk=args.chunk, out_dir=Path(args.out), tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

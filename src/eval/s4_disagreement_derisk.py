"""S4 de-risk — does geometric disagreement separate clean vs. corrupted goal patches?

The core S4 hypothesis, tested BEFORE building the bridge (M1) and BEFORE any big materialize:
native-encode a clip ``[obs … goal]`` through the *frozen* backbone (the S2 corruptions are edits
of real frames, so they encode fine), compute a per-patch **cycle error** on the goal frame, and
ask whether it is higher in the corrupted patches than the clean ones — AUROC vs. the S2 label.

- Holds  → the disagreement signal is real; M1's job is just to make *dreamed* frames behave like this.
- Fails  → rethink before spending on M1.

Scores per patch (16×16): absolute cycle error on the corrupted clip; the clean→corrupt **delta**;
and (1 − visibility). Localized corruptions (cut-paste / tps / composite) drive the per-patch AUROC;
whole-frame `mismatch` drives a separate global check.

Run: see experiments/S4_disagreement_derisk/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))                          # our src/ for `import data`

import s0_backbone_sanity as s0                        # noqa: E402 (co-located; self-inserts vendored repo)
from data import WindowConfig, build_tuple, plan_tuples  # noqa: E402
from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402
from data.corruptions import (  # noqa: E402
    GRID, cross_episode_composite, cut_paste, mismatched_frame, tps_warp,
)
from data.sam2_masks import Sam2MaskGenerator  # noqa: E402

LOCAL_TYPES = ["remove", "relocate", "duplicate", "tps", "composite"]


# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks_sorted = np.arange(1, len(a) + 1, dtype=float)
    i = 0
    while i < len(a):                                   # average ranks over ties
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    ranks = np.empty(len(a))
    ranks[order] = ranks_sorted
    return ranks


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    P, N = int(label.sum()), int((~label).sum())
    if P == 0 or N == 0:
        return float("nan")
    r = _rankdata(score)
    return float((r[label].sum() - P * (P + 1) / 2) / (P * N))


# ---------------------------------------------------------------------------
def build_clip(tup, size=256, K=16) -> np.ndarray:
    """[obs … waypoints … goal] → K uniformly-sampled frames (goal forced last), resized to `size`."""
    frames = np.concatenate([tup.obs_frames, tup.waypoint_frames], axis=0)   # last = goal (t+n)
    T = frames.shape[0]
    idx = np.linspace(0, T - 1, K).round().astype(int)
    idx[-1] = T - 1
    idx = np.unique(idx)
    return np.stack([cv2.resize(np.ascontiguousarray(frames[i]), (size, size),
                                interpolation=cv2.INTER_LINEAR) for i in idx])


def make_corruption(ctype, goal, masks, donor, donor_masks, rng):
    if ctype == "remove":
        return cut_paste(goal, masks[0], "remove")
    if ctype == "relocate":
        return cut_paste(goal, masks[0], "relocate", rng)
    if ctype == "duplicate":
        return cut_paste(goal, masks[min(1, len(masks) - 1)], "duplicate", rng)
    if ctype == "tps":
        return tps_warp(goal, masks[0], rng=rng)
    if ctype == "composite":
        return cross_episode_composite(goal, donor, donor_masks[0], rng)
    return mismatched_frame(goal, donor)               # mismatch


def cyc_vis_per_patch(model, clip_u8, query_grid=32, chunk=4096):
    """Per-patch (16×16) mean cycle error (px) and mean (1−visibility), from goal→obs0→goal."""
    device = next(model.parameters()).device
    K, H, W, _ = clip_u8.shape
    video_b = (torch.from_numpy(clip_u8).to(device=device, dtype=torch.float32)
               .permute(0, 3, 1, 2).unsqueeze(0) / 255.0)
    aspect_b = torch.tensor([[float(W) / float(H)]], dtype=torch.float32, device=device)
    with torch.no_grad():
        mem = s0._encode_model_memory(model=model, video_b=video_b, aspect_b=aspect_b)
        g, m = query_grid, 0.02
        xs = np.linspace(m, 1 - m, g, dtype=np.float32)
        ys = np.linspace(m, 1 - m, g, dtype=np.float32)
        uv0 = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
        n = uv0.shape[0]
        last = np.full((n,), K - 1, np.int64)
        z = np.zeros((n,), np.int64)
        fwd = s0._query(model, video_b, aspect_b, mem, uv0, last, z, z, chunk)      # goal → obs0
        uvf = fwd["uv_2d"]
        vis = s0._sigmoid(fwd["visibility"])
        bwd = s0._query(model, video_b, aspect_b, mem, uvf, z, last, last, chunk)    # obs0 → goal
        cyc = np.linalg.norm((bwd["uv_2d"] - uv0) * np.array([W - 1, H - 1], np.float32), axis=1)

    pj = np.clip((uv0[:, 0] * GRID).astype(int), 0, GRID - 1)
    pi = np.clip((uv0[:, 1] * GRID).astype(int), 0, GRID - 1)
    cyc_map = np.zeros((GRID, GRID)); invis_map = np.zeros((GRID, GRID)); cnt = np.zeros((GRID, GRID))
    np.add.at(cyc_map, (pi, pj), cyc)
    np.add.at(invis_map, (pi, pj), 1.0 - vis)
    np.add.at(cnt, (pi, pj), 1.0)
    cnt = np.maximum(cnt, 1.0)
    return cyc_map / cnt, invis_map / cnt


def gather_tuples(n_files, per_file, cfg):
    files = sorted((LIBERO_ROOT / "libero_spatial").glob("*.hdf5"))
    out = []
    for f in files[:n_files]:
        traj = LiberoTrajectory(f, 0)
        specs = plan_tuples(traj.episode_id, len(traj), cfg)
        for sp in specs[:per_file]:
            out.append(build_tuple(traj, sp))
    return out, files


def main() -> int:
    ap = argparse.ArgumentParser(description="S4 disagreement de-risk (no bridge).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-files", type=int, default=7)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(_SRC.parents[0] / "experiments" / "S4_disagreement_derisk"))
    args = ap.parse_args()

    device = s0._resolve_device(args.device)
    model = s0.load_backbone(args.config, args.ckpt, device)
    mg = Sam2MaskGenerator(device=("cuda" if device.type == "cuda" else "cpu"))
    cfg = WindowConfig()
    rng = np.random.default_rng(0)

    tuples, files = gather_tuples(args.n_files, args.per_file, cfg)
    donor_clip = build_clip(build_tuple(LiberoTrajectory(files[-1], 0),
                                        plan_tuples("d", len(LiberoTrajectory(files[-1], 0)), cfg)[0]))
    donor = donor_clip[-1]
    donor_masks = mg.object_masks(donor)
    print(f"tuples={len(tuples)}  donor_masks={len(donor_masks)}")

    local = {"cyc": [], "delta": [], "invis": [], "label": []}
    mism = []
    examples = []
    for i, tup in enumerate(tuples):
        clip = build_clip(tup)
        goal = clip[-1]
        masks = mg.object_masks(goal)
        if not masks or not donor_masks:
            continue
        ctype = LOCAL_TYPES[i % len(LOCAL_TYPES)] if (i % 5 != 4) else "mismatch"
        corr = make_corruption(ctype, goal, masks, donor, donor_masks, rng)
        cclip = clip.copy(); cclip[-1] = corr.frame
        cyc_c, invis_c = cyc_vis_per_patch(model, cclip)
        cyc_0, _ = cyc_vis_per_patch(model, clip)
        if ctype == "mismatch":
            mism.append((float(cyc_0.mean()), float(cyc_c.mean())))
            continue
        lab = corr.patch_mask
        local["cyc"].append(cyc_c.ravel()); local["delta"].append((cyc_c - cyc_0).ravel())
        local["invis"].append(invis_c.ravel()); local["label"].append(lab.ravel())
        if len(examples) < 3:
            examples.append((corr.frame, np.clip(cyc_c - cyc_0, 0, None), lab, ctype))
        print(f"  [{i}] {ctype:9s} label={int(lab.sum()):3d}  cyc_corrupt={cyc_c.mean():.2f}px")

    for k in ("cyc", "delta", "invis", "label"):
        local[k] = np.concatenate(local[k]) if local[k] else np.array([])
    report = {
        "n_localized_tuples": int(len(local["label"]) // (GRID * GRID)) if local["label"].size else 0,
        "auroc_cycle_error": auroc(local["cyc"], local["label"]),
        "auroc_cycle_delta": auroc(local["delta"], local["label"]),
        "auroc_invisibility": auroc(local["invis"], local["label"]),
        "mismatch_global": {
            "n": len(mism),
            "mean_cyc_clean": float(np.mean([a for a, _ in mism])) if mism else None,
            "mean_cyc_corrupt": float(np.mean([b for _, b in mism])) if mism else None,
        },
    }
    out = Path(args.out); (out / "viz").mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    _save_examples(examples, out / "viz" / "s4_derisk_examples.png")
    _save_roc(local, out / "viz" / "s4_derisk_roc.png")
    _save_distribution(local, out / "viz" / "s4_derisk_distribution.png")
    print(json.dumps(report, indent=2))
    return 0


def _save_examples(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)), dpi=120)
    axes = np.atleast_2d(axes)
    for r, (frame, cyc, lab, ctype) in enumerate(examples):
        axes[r, 0].imshow(frame); axes[r, 0].set_ylabel(ctype, fontsize=10)
        im = axes[r, 1].imshow(cyc, cmap="magma"); fig.colorbar(im, ax=axes[r, 1], fraction=0.046)
        axes[r, 1].contour(lab.astype(float), levels=[0.5], colors="cyan", linewidths=1.3)  # label outline
        axes[r, 2].imshow(lab, cmap="Greys_r")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            for c, t in enumerate(["corrupted goal", "cycle-error RISE vs clean (px)", "S2 label"]):
                axes[r, c].set_title(t, fontsize=11)
    fig.suptitle("S4 de-risk: the clean→corrupt cycle-error DELTA localizes the corruption\n"
                 "(cyan outline = S2 label)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _roc_points(score, label):
    label = label.astype(bool)
    order = np.argsort(-score, kind="mergesort")
    l = label[order]
    tpr = np.concatenate([[0.0], np.cumsum(l) / max(label.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~l) / max((~label).sum(), 1)])
    return fpr, tpr


def _save_roc(local, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.3), dpi=120)
    for key, name, col in [("delta", "cycle-error delta (corrupt−clean)", "#e45756"),
                           ("invis", "1 − visibility", "#54a24b"),
                           ("cyc", "absolute cycle error", "#4c78a8")]:
        fpr, tpr = _roc_points(local[key], local["label"])
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{name}   AUROC {auroc(local[key], local['label']):.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_aspect("equal"); ax.set_title("S4 de-risk — do corrupted patches rank above clean?")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _save_distribution(local, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, lab = local["delta"], local["label"].astype(bool)
    bins = np.linspace(float(np.percentile(d, 1)), float(np.percentile(d, 99)), 40)
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=120)
    ax.hist(d[~lab], bins=bins, density=True, alpha=0.6, color="#4c78a8", label=f"clean patches (n={int((~lab).sum())})")
    ax.hist(d[lab], bins=bins, density=True, alpha=0.6, color="#e45756", label=f"corrupted patches (n={int(lab.sum())})")
    ax.axvline(np.median(d[~lab]), color="#4c78a8", ls="--", lw=1.5)
    ax.axvline(np.median(d[lab]), color="#e45756", ls="--", lw=1.5)
    ax.set_xlabel("cycle-error rise vs clean (px)"); ax.set_ylabel("density")
    ax.set_title("Corrupted patches shift to higher disagreement\n"
                 f"median: clean {np.median(d[~lab]):.2f}px → corrupted {np.median(d[lab]):.2f}px")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

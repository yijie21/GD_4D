"""M4 v2 — the real Dₜ head: a per-patch semantic critic that catches WITHIN-SCENE
wrong-object errors and generalizes across LIBERO suites.

M4 (CLS) showed a learned semantic critic separates correct/wrong goals (0.83-0.98)
where geometry couldn't, and transfers to generated dreams (scene-fit 0.78). The open
caveat was WITHIN-SCENE fine-grained wrongness (an object in the wrong place, same
scene) — untestable on homogeneous libero_spatial.

This closes it with the plan's S4-supervised head on DINOv2 PATCH features:
  * genuinely-wrong same-scene negatives = S2 corruptions (relocate / remove / tps) of
    the real goal (SAM2 object masks) with 16×16 patch labels.
  * per-patch feature = [dino_patch(goal), dino_patch(goal) − dino_patch(o_t)] — the
    goal patch and how it changed vs the observation.
  * label = corrupted patch (1) vs clean (0). Clean goals contribute all-0, so the
    critic must separate a CORRUPTION-change from a LEGITIMATE motion-change (arm /
    the object legitimately moving) — exactly what geometry (M1/M2/M3) could not do.

Evaluated two ways:
  * within-distribution: held-out demos, all 4 suites.
  * cross-suite OOD: train on 3 suites, test on the held-out suite (unseen scenes).
Baseline: 1 − cos(goal_patch, o_t_patch) with no training (does raw feature-change
detect corruption, or does it fire on legit motion too?).

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    export HF_HOME=/workspace/huggingface_cache/
    CUDA_VISIBLE_DEVICES=0 python experiments/M4_learned_critic/critic_v2_patch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC = Path("/workspace/code/GD_4D/src")
sys.path.insert(0, str(_SRC))
from data.adapters import LIBERO_ROOT, LiberoTrajectory  # noqa: E402
from data.corruptions import GRID, cut_paste, tps_warp, region_to_patch_labels  # noqa: E402
from data.sam2_masks import Sam2MaskGenerator  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
MODEL_ID = "facebook/dinov2-base"
SIZE = 256
H = 20


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    P, N = int(label.sum()), int((~label).sum())
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]; i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[label].sum() - P * (P + 1) / 2) / (P * N))


class DinoPatch:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(device).eval().to(torch.float16)
        self.device = device

    @torch.no_grad()
    def patches(self, imgs_u8, bs=48):
        """list of [H,W,3] uint8 -> [N, GRID, GRID, 768] float32 patch features."""
        from PIL import Image
        out = []
        for i in range(0, len(imgs_u8), bs):
            batch = [Image.fromarray(x) for x in imgs_u8[i:i + bs]]
            inp = self.proc(images=batch, return_tensors="pt").to(self.device)
            inp = {k: v.to(torch.float16) for k, v in inp.items()}
            tok = self.model(**inp).last_hidden_state[:, 1:]        # drop CLS -> [B,256,768]
            g = int(round(tok.shape[1] ** 0.5))
            tok = tok.reshape(tok.shape[0], g, g, tok.shape[-1])
            if g != GRID:
                tok = torch.nn.functional.interpolate(
                    tok.permute(0, 3, 1, 2).float(), size=(GRID, GRID), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
            out.append(tok.float().cpu().numpy())
        return np.concatenate(out)


def _resize(f):
    return cv2.resize(np.ascontiguousarray(f), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def make_corruption(goal, masks, rng, i):
    ctype = ["relocate", "remove", "tps"][i % 3]
    if ctype == "relocate":
        return cut_paste(goal, masks[0], "relocate", rng), ctype
    if ctype == "remove":
        return cut_paste(goal, masks[0], "remove"), ctype
    return tps_warp(goal, masks[0], rng=rng), ctype


def build_windows(per_suite_tasks=6, demos=4):
    """Lazy: (suite, o_t, clean_goal) windows across all 4 suites."""
    rows = []
    for suite in SUITES:
        files = sorted((LIBERO_ROOT / suite).glob("*.hdf5"))[:per_suite_tasks]
        for f in files:
            for di in range(demos):
                try:
                    tr = LiberoTrajectory(f, di)
                except (IndexError, FileNotFoundError):
                    continue
                T = len(tr)
                if T < H + 4:
                    continue
                t = T // 2
                g = min(t + H, T - 1)
                rows.append({"suite": suite, "obs": _resize(tr.frame(t)),
                             "goal": _resize(tr.frame(g)), "demo": di})
    return rows


def train_eval(Xtr, ytr, Xte, yte, epochs=250, lr=1e-3, wd=1e-3, hidden=256):
    dev = "cuda"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ (ytr == 0).sum() / max((ytr == 1).sum(), 1) ], device=dev))
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xt).squeeze(1), yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s = net(torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)).squeeze(1).cpu().numpy()
    return auroc(s, yte)


def main():
    device = "cuda"
    rng = np.random.default_rng(0)
    print("building windows across suites ...", flush=True)
    rows = build_windows()
    print(f"windows: {len(rows)}  per-suite: " +
          ", ".join(f"{s}={sum(r['suite']==s for r in rows)}" for s in SUITES))

    dino = DinoPatch(device)
    mg = Sam2MaskGenerator(device=device)

    # per-window: build corrupted goal + labels, embed patches
    samples = []  # dict: suite, feat[GRID*GRID, 1536], label[GRID*GRID]
    kept = 0
    for i, r in enumerate(rows):
        masks = mg.object_masks(r["goal"])
        if not masks:
            continue
        corr, ctype = make_corruption(r["goal"], masks, rng, i)
        P = dino.patches([r["obs"], r["goal"], corr.frame])  # [3,G,G,768]
        po, pg_clean, pg_corr = P[0], P[1], P[2]
        feat_clean = np.concatenate([pg_clean, pg_clean - po], axis=-1).reshape(GRID * GRID, -1)
        feat_corr = np.concatenate([pg_corr, pg_corr - po], axis=-1).reshape(GRID * GRID, -1)
        lab_clean = np.zeros(GRID * GRID)
        lab_corr = corr.patch_mask.reshape(GRID * GRID).astype(float)
        samples.append({"suite": r["suite"], "demo": r["demo"],
                        "feat": np.concatenate([feat_clean, feat_corr]),
                        "label": np.concatenate([lab_clean, lab_corr]),
                        "po": np.concatenate([po.reshape(GRID * GRID, -1)] * 2),
                        "pg": np.concatenate([pg_clean.reshape(GRID * GRID, -1),
                                              pg_corr.reshape(GRID * GRID, -1)])})
        kept += 1
        if kept % 20 == 0:
            print(f"  {kept} windows corrupted+embedded", flush=True)

    report = {"n_windows": kept}

    def stack(sel):
        X = np.concatenate([s["feat"] for s in sel]); y = np.concatenate([s["label"] for s in sel])
        po = np.concatenate([s["po"] for s in sel]); pg = np.concatenate([s["pg"] for s in sel])
        return X, y, po, pg

    # cosine baseline (no training): 1 - cos(goal_patch, obs_patch)
    def cos_base(po, pg, y):
        c = (po * pg).sum(1) / (np.linalg.norm(po, axis=1) * np.linalg.norm(pg, axis=1) + 1e-6)
        return auroc(1 - c, y)

    # (A) within-distribution: split by demo index
    tr = [s for s in samples if s["demo"] < 3]
    te = [s for s in samples if s["demo"] >= 3]
    Xtr, ytr, _, _ = stack(tr); Xte, yte, po_te, pg_te = stack(te)
    report["within_dist_probe_auroc"] = round(train_eval(Xtr, ytr, Xte, yte), 3)
    report["within_dist_cosine_auroc"] = round(cos_base(po_te, pg_te, yte), 3)
    print(f"\n[within-distribution]  probe AUROC {report['within_dist_probe_auroc']}   "
          f"cosine-baseline {report['within_dist_cosine_auroc']}  (per-patch, {len(yte)} patches)")

    # (B) cross-suite OOD: train on 3 suites, test on the held-out suite
    report["ood"] = {}
    for held in SUITES:
        tr = [s for s in samples if s["suite"] != held]
        te = [s for s in samples if s["suite"] == held]
        if not te or not tr:
            continue
        Xtr, ytr, _, _ = stack(tr); Xte, yte, po_te, pg_te = stack(te)
        a = train_eval(Xtr, ytr, Xte, yte)
        report["ood"][held] = {"probe": round(a, 3), "cosine": round(cos_base(po_te, pg_te, yte), 3),
                               "n_patches": int(len(yte)), "n_corrupt": int(yte.sum())}
        print(f"[OOD held-out {held:15s}]  probe AUROC {a:.3f}   "
              f"cosine {report['ood'][held]['cosine']:.3f}  ({int(yte.sum())} corrupt / {len(yte)} patches)")

    Path(__file__).resolve().parent.joinpath("v2_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[save] v2_report.json")
    print("Read-out: probe >> cosine and >> M3 geometry (~0.6) with OOD holding up => a learned "
          "patch critic detects within-scene wrong objects and generalizes across scenes.")


if __name__ == "__main__":
    main()

"""M4 learned-critic de-risk — Q1: can a semantic-feature critic judge goal-scene fit?

M3 showed hand-designed GEOMETRIC signals over the frozen backbone can't tell a
correct goal from a wrong one (only broken images). This asks whether a small
LEARNED critic over SEMANTIC features (DINOv2) can — the thing the disagreement
head would actually be.

Q1 (capability, this script): separate a scene's CORRECT goal from a WRONG goal,
using DINOv2 embeddings of (o_t, goal). Positives and negatives are BOTH real
frames sharing the SAME o_t, so there is no real-vs-fake or obs shortcut — the
critic must use the *relationship* between o_t and the goal.

Negatives (reference-free, genuinely wrong for THIS scene):
  * diff_task : real goal from a DIFFERENT task           (wrong scene/objects — clear)
  * same_task : real goal from a DIFFERENT demo, same task (ambiguous: arguably a valid goal)

Feature per (o_t, goal): [emb(o_t), emb(goal), emb(goal)-emb(o_t)] (DINOv2 CLS, 768*3).
Probe: logistic regression (torch), trained on train demos, evaluated on held-out demos.
Baselines: cosine(o_t, goal) with no training.

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    export HF_HOME=/workspace/huggingface_cache/
    CUDA_VISIBLE_DEVICES=0 python experiments/M4_learned_critic/critic_derisk.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "M2_libero_dreamer"))
from data import load_libero_spatial  # noqa: E402  (M2 loader)

MODEL_ID = "facebook/dinov2-base"


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    P, N = int(label.sum()), int((~label).sum())
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    s_sorted = score[order]; i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[label].sum() - P * (P + 1) / 2) / (P * N))


class Dino:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(device).eval().to(torch.float16)
        self.device = device

    @torch.no_grad()
    def embed(self, imgs_u8, bs=64):
        """list/array of [H,W,3] uint8 -> [N,768] float32 CLS embeddings."""
        from PIL import Image
        out = []
        for i in range(0, len(imgs_u8), bs):
            batch = [Image.fromarray(x) for x in imgs_u8[i:i + bs]]
            inp = self.proc(images=batch, return_tensors="pt").to(self.device)
            inp = {k: v.to(torch.float16) for k, v in inp.items()}
            cls = self.model(**inp).last_hidden_state[:, 0]  # CLS
            out.append(cls.float().cpu().numpy())
        return np.concatenate(out)


def build_windows(tasks, H=20, val_demos=10):
    """Per (task, demo): o_t and correct goal, plus indices for wrong goals."""
    rows = []
    for ti, t in enumerate(tasks):
        n = len(t["demos"])
        for di in range(n):
            fr = t["demos"][di]
            T = fr.shape[0]
            if T < H + 4:
                continue
            split = "val" if di >= n - val_demos else "train"
            for frac in (0.30, 0.50, 0.70):
                tt = int(T * frac)
                g = min(tt + H, T - 1)
                rows.append({"ti": ti, "di": di, "t": tt, "g": g, "split": split})
    return rows


def train_probe(Xtr, ytr, hidden=256, epochs=400, lr=1e-3, wd=1e-2):
    """Small MLP probe (subsumes linear; can use nonlinear combos like diff-norm)."""
    dev = "cuda"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt).squeeze(1), yt)
        loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


def score_probe(net, mu, sd, X):
    with torch.no_grad():
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device="cuda")
        return net(Xt).squeeze(1).cpu().numpy()


def main():
    device = "cuda"
    tasks = load_libero_spatial()
    rows = build_windows(tasks)
    print(f"windows: {len(rows)}  ({sum(r['split']=='train' for r in rows)} train / "
          f"{sum(r['split']=='val' for r in rows)} val)")

    dino = Dino(device)
    # embed every needed frame once: o_t and goal of each row
    obs_imgs = [tasks[r["ti"]]["demos"][r["di"]][r["t"]] for r in rows]
    goal_imgs = [tasks[r["ti"]]["demos"][r["di"]][r["g"]] for r in rows]
    print("embedding obs + goals ...", flush=True)
    E_obs = dino.embed(obs_imgs)
    E_goal = dino.embed(goal_imgs)
    # index by (ti,di) -> goal embedding, for wrong-goal lookup
    key2goal = {(r["ti"], r["di"]): E_goal[i] for i, r in enumerate(rows)}
    tasks_by = {}
    for i, r in enumerate(rows):
        tasks_by.setdefault(r["ti"], []).append((r["di"], i))
    rng = np.random.default_rng(0)

    def feat(eo, eg):
        return np.concatenate([eo, eg, eg - eo])

    # assemble pos + negatives per split
    def assemble(split):
        pos, neg_same, neg_diff = [], [], []
        idxs = [i for i, r in enumerate(rows) if r["split"] == split]
        n_tasks = len(tasks)
        for i in idxs:
            r = rows[i]
            eo = E_obs[i]
            pos.append(feat(eo, E_goal[i]))
            # same-task, different demo
            cand = [j for dd, j in tasks_by[r["ti"]] if j != i]
            if cand:
                j = cand[int(rng.integers(len(cand)))]
                neg_same.append(feat(eo, E_goal[j]))
            # different task
            ot = (r["ti"] + 1 + int(rng.integers(n_tasks - 1))) % n_tasks
            if ot in tasks_by and tasks_by[ot]:
                dd, j = tasks_by[ot][int(rng.integers(len(tasks_by[ot])))]
                neg_diff.append(feat(eo, E_goal[j]))
        return np.array(pos), np.array(neg_same), np.array(neg_diff)

    Ptr, Ns_tr, Nd_tr = assemble("train")
    Pte, Ns_te, Nd_te = assemble("val")

    report = {"n_windows": len(rows)}
    for name, Ntr, Nte in [("diff_task", Nd_tr, Nd_te), ("same_task", Ns_tr, Ns_te)]:
        Xtr = np.concatenate([Ptr, Ntr]); ytr = np.concatenate([np.ones(len(Ptr)), np.zeros(len(Ntr))])
        Xte = np.concatenate([Pte, Nte]); yte = np.concatenate([np.ones(len(Pte)), np.zeros(len(Nte))])
        net, mu, sd = train_probe(Xtr, ytr)
        a = auroc(score_probe(net, mu, sd, Xte), yte)
        # cosine baseline (no training): correct goal should be more obs-consistent?
        def cos_scores(P, N):
            # use goal-vs-obs cosine as "consistency"; higher = more consistent (label 1)
            def c(F):
                eo, eg = F[:, :768], F[:, 768:1536]
                return (eo * eg).sum(1) / (np.linalg.norm(eo, axis=1) * np.linalg.norm(eg, axis=1) + 1e-6)
            s = np.concatenate([c(P), c(N)])
            y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
            return auroc(s, y)
        cos_a = cos_scores(Pte, Nte)
        report[f"auroc_probe__{name}"] = round(float(a), 3)
        report[f"auroc_cosine__{name}"] = round(float(cos_a), 3)
        print(f"  {name:10s}: probe AUROC {a:.3f}   cosine-baseline {cos_a:.3f}  "
              f"(train {len(Xtr)}, test {len(Xte)})")

    # ------------------------------------------------------------------ Q2
    # OOD gate: does the critic (trained on REAL frames) transfer to GENERATED
    # dreams? Train on ALL real negatives; evaluate on M2 dreams by pairing each
    # obs with its own dream (correct) vs another item's dream (mismatched scene).
    dreams_p = _HERE.parent / "M2_libero_dreamer" / "dreams" / "dreams.npz"
    if dreams_p.exists():
        Xtr = np.concatenate([Ptr, Ns_tr, Nd_tr])
        ytr = np.concatenate([np.ones(len(Ptr)), np.zeros(len(Ns_tr) + len(Nd_tr))])
        net, mu, sd = train_probe(Xtr, ytr)

        d = np.load(dreams_p, allow_pickle=True)
        obs_d, dream_d, goal_d = d["obs"], d["dream"], d["goal"]
        Eod = dino.embed(list(obs_d)); Edd = dino.embed(list(dream_d)); Egd = dino.embed(list(goal_d))
        N = len(obs_d)

        # (a) correct pairing (obs_i, dream_i) vs mismatched (obs_i, dream_j)
        pos = np.array([feat(Eod[i], Edd[i]) for i in range(N)])
        neg = np.array([feat(Eod[i], Edd[j]) for i in range(N)
                        for j in range(N) if j != i])
        s = score_probe(net, mu, sd, np.concatenate([pos, neg]))
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        auroc_pairing = auroc(s, y)

        # (b) does the critic prefer the REAL goal over the dream for the SAME obs?
        s_real = score_probe(net, mu, sd, np.array([feat(Eod[i], Egd[i]) for i in range(N)]))
        s_dream = score_probe(net, mu, sd, np.array([feat(Eod[i], Edd[i]) for i in range(N)]))
        real_pref = float((s_real > s_dream).mean())

        report["q2_auroc_dream_scene_pairing"] = round(float(auroc_pairing), 3)
        report["q2_realgoal_preferred_over_dream"] = round(real_pref, 3)
        print(f"\n  [Q2 transfer to GENERATED dreams]")
        print(f"    dream-fits-this-scene (obs_i,dream_i vs obs_i,dream_j): AUROC {auroc_pairing:.3f}")
        print(f"    real goal scored above dream (same obs): {real_pref:.2f} of {N}")

    (_HERE / "q1_report.json").write_text(json.dumps(report, indent=2))
    print(f"[save] {_HERE/'q1_report.json'}")


if __name__ == "__main__":
    main()

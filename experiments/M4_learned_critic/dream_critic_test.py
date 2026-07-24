"""Close the synthetic-corruption gap: does the critic catch a REAL dreamer's errors?

Uses the 4-suite dreamer's correct- vs wrong-instruction dreams (dreams_cw.npz).
Both dreams are generated and share the scene; only the goal's correctness differs.

Q3a (ceiling): can a critic separate dream_correct from dream_wrong at all?
    -> train on dreams from 3 suites, test on the held-out suite (CLS features).
Q3b (transfer / the §5.2 OOD gate): does a critic trained ONLY on CHEAP real-frame
    negatives (obs+correct-real-goal vs obs+wrong-task-real-goal) — never a dream —
    flag the real dream errors?  This is the deployment question: train on cheap
    data, detect real dream failures.

Reproduce (env gd4d5090, GPU0):
    cd /workspace/code/GD_4D
    export HF_HOME=/workspace/huggingface_cache/
    CUDA_VISIBLE_DEVICES=0 python experiments/M4_learned_critic/dream_critic_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "M2_libero_dreamer"))
from critic_derisk import Dino, train_probe, score_probe, auroc  # noqa: E402
from data import load_libero  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DREAMS = _HERE.parent / "M2_libero_dreamer" / "dreams_cw" / "dreams_cw.npz"
MODEL_ID_IP2P = "timbrooks/instruct-pix2pix"


def feat(eo, eg):
    return np.concatenate([eo, eg, eg - eo], axis=-1)


def build_real(dino, per_task=4, per_suite_tasks=8, H=30):
    """Cheap real-frame training set: (o_t, correct goal)=1 vs (o_t, wrong-task goal)=0."""
    tasks = load_libero(SUITES, max_demos=per_task + 2)
    obs_imgs, goal_imgs, tkey = [], [], []
    for ti, t in enumerate(tasks):
        for di in range(min(per_task, len(t["demos"]))):
            fr = t["demos"][di]; T = fr.shape[0]
            tt = T // 2; g = min(tt + H, T - 1)
            obs_imgs.append(np.asarray(Image.fromarray(fr[tt]).resize((256, 256))))
            goal_imgs.append(np.asarray(Image.fromarray(fr[g]).resize((256, 256))))
            tkey.append(ti)
    Eo = dino.embed(obs_imgs); Eg = dino.embed(goal_imgs)
    tkey = np.array(tkey)
    rng = np.random.default_rng(0)
    pos, neg = [], []
    for i in range(len(Eo)):
        pos.append(feat(Eo[i], Eg[i]))
        # wrong goal: a goal from a different task
        cand = np.where(tkey != tkey[i])[0]
        j = cand[int(rng.integers(len(cand)))]
        neg.append(feat(Eo[i], Eg[j]))
    X = np.concatenate([np.array(pos), np.array(neg)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return X, y


def main():
    if not DREAMS.exists():
        raise SystemExit(f"missing {DREAMS} — run gen_correct_wrong.py first (needs the 4-suite dreamer).")
    dino = Dino("cuda")

    d = np.load(DREAMS, allow_pickle=True)
    obs, dc, dw, suite, instr = d["obs"], d["dream_correct"], d["dream_wrong"], d["suite"], d["instr"]
    N = len(obs)
    Eo = dino.embed(list(obs)); Ec = dino.embed(list(dc)); Ew = dino.embed(list(dw))
    Fc = feat(Eo, Ec); Fw = feat(Eo, Ew)   # correct dream (1) / wrong dream (0)
    print(f"dreams: {N} items  per-suite: " + ", ".join(f"{s}={int((suite==s).sum())}" for s in SUITES))

    # CLIP text embedding of the (correct, intended) instruction — the critic knows
    # what the goal SHOULD achieve; both dreams are judged against the SAME instruction.
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(MODEL_ID_IP2P, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(MODEL_ID_IP2P, subfolder="text_encoder").to("cuda").eval()
    with torch.no_grad():
        ids = tok(list(instr), padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to("cuda")
        Etxt = te(ids).pooler_output.float().cpu().numpy()   # [N,768]
    Fc_t = np.concatenate([Fc, Etxt], axis=1); Fw_t = np.concatenate([Fw, Etxt], axis=1)

    report = {"n_items": int(N)}

    # ---- Q3a: ceiling, train on 3 suites' dreams, test held-out suite ----------
    report["q3a_ceiling_ood"] = {}
    for held in SUITES:
        tr = suite != held; te = suite == held
        if te.sum() == 0:
            continue
        Xtr = np.concatenate([Fc[tr], Fw[tr]]); ytr = np.concatenate([np.ones(tr.sum()), np.zeros(tr.sum())])
        Xte = np.concatenate([Fc[te], Fw[te]]); yte = np.concatenate([np.ones(te.sum()), np.zeros(te.sum())])
        net, mu, sd = train_probe(Xtr, ytr)
        a = auroc(score_probe(net, mu, sd, Xte), yte)
        report["q3a_ceiling_ood"][held] = round(float(a), 3)
        print(f"  [Q3a ceiling, held-out {held:15s}] correct-vs-wrong dream AUROC {a:.3f}  (test {int(te.sum())}x2)")

    # ---- Q3a-indist: in-distribution ceiling (random split, not suite-OOD) ------
    rng = np.random.default_rng(0)
    perm = rng.permutation(N); cut = int(0.6 * N)
    tri, tei = perm[:cut], perm[cut:]
    for tag, Fcx, Fwx in [("appearance", Fc, Fw), ("+instruction", Fc_t, Fw_t)]:
        Xtr = np.concatenate([Fcx[tri], Fwx[tri]]); ytr = np.concatenate([np.ones(len(tri)), np.zeros(len(tri))])
        Xte = np.concatenate([Fcx[tei], Fwx[tei]]); yte = np.concatenate([np.ones(len(tei)), np.zeros(len(tei))])
        net, mu, sd = train_probe(Xtr, ytr)
        a = auroc(score_probe(net, mu, sd, Xte), yte)
        report[f"q3a_indist_{tag.strip('+')}"] = round(float(a), 3)
        print(f"  [Q3a in-distribution ceiling, {tag:12s}] AUROC {a:.3f}")

    # ---- Q3b: transfer — critic trained ONLY on cheap real frames --------------
    Xr, yr = build_real(dino)
    net, mu, sd = train_probe(Xr, yr)
    s = score_probe(net, mu, sd, np.concatenate([Fc, Fw]))
    y = np.concatenate([np.ones(N), np.zeros(N)])
    a_transfer = auroc(s, y)
    pref = float((score_probe(net, mu, sd, Fc) > score_probe(net, mu, sd, Fw)).mean())
    report["q3b_transfer_auroc"] = round(float(a_transfer), 3)
    report["q3b_correct_preferred"] = round(pref, 3)
    print(f"\n  [Q3b transfer] critic trained ONLY on cheap real-frame negatives:")
    print(f"    real dream correct-vs-wrong AUROC {a_transfer:.3f}   correct preferred {pref:.2f} of {N}")

    # ---- Q3c: WITH the instruction (CLIP text) — is the mismatch detectable now? --
    report["q3c_with_instruction_ood"] = {}
    for held in SUITES:
        tr = suite != held; te_m = suite == held
        if te_m.sum() == 0:
            continue
        Xtr = np.concatenate([Fc_t[tr], Fw_t[tr]]); ytr = np.concatenate([np.ones(tr.sum()), np.zeros(tr.sum())])
        Xte = np.concatenate([Fc_t[te_m], Fw_t[te_m]]); yte = np.concatenate([np.ones(te_m.sum()), np.zeros(te_m.sum())])
        net, mu, sd = train_probe(Xtr, ytr)
        a = auroc(score_probe(net, mu, sd, Xte), yte)
        report["q3c_with_instruction_ood"][held] = round(float(a), 3)
        print(f"  [Q3c +instruction, held-out {held:15s}] AUROC {a:.3f}")
    q3a = np.mean([v for v in report["q3a_ceiling_ood"].values()])
    q3c = np.mean([v for v in report["q3c_with_instruction_ood"].values()])
    print(f"\n  mean OOD AUROC:  appearance-only (Q3a) {q3a:.3f}   +instruction (Q3c) {q3c:.3f}")

    (_HERE / "dream_critic_report.json").write_text(json.dumps(report, indent=2))
    print(f"[save] dream_critic_report.json")


if __name__ == "__main__":
    main()

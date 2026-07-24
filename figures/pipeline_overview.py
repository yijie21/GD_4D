"""GD-4D full pipeline — one self-explanatory overview figure.

Color-codes every stage by its de-risk status so the figure doubles as a
project status board: what is validated, what is built-but-open, what is
FALSIFIED (the disagreement head Dt), and what is not started.

Uses scipilot-figure-skill's setup_style for publication fonts (graceful
fallback if unavailable). This is a schematic (boxes+arrows), which scipilot
does not "advise" on — only its styling/export infra is borrowed.

Reproduce (env gd4d5090):
    cd /workspace/code/GD_4D
    python figures/pipeline_overview.py
    # -> figures/fig0_pipeline_overview.{png,svg,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Optional: scipilot publication styling.
_SK = Path("/home/yijie/.claude/skills/scipilot-figure-skill/scripts")
if _SK.exists():
    sys.path.insert(0, str(_SK))
try:
    from setup_style import setup_style
    setup_style(journal="general", lang="en")
except Exception:
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})

# Okabe-Ito colorblind-safe status palette
C = {
    "validated": "#009E73",   # green
    "open":      "#E69F00",   # amber - built but question open
    "resolved":  "#0072B2",   # blue - de-risked, redefined -> build
    "falsified": "#D55E00",   # vermillion - falsified idea (shown in callout only)
    "todo":      "#999999",   # gray - not started
    "input":     "#56B4E9",   # blue - data
}
FILL = {k: v for k, v in C.items()}


def tint(hexc, a=0.13):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hexc)
    return (r, g, b, a)


def box(ax, x, y, w, h, title, subtitle="", status="todo", stage="", fs_title=10.5):
    col = C[status]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                       linewidth=2.4, edgecolor=col, facecolor=tint(col),
                       mutation_aspect=1, zorder=3)
    ax.add_patch(p)
    cx = x + w / 2
    ax.text(cx, y + h - h * 0.34, title, ha="center", va="center",
            fontsize=fs_title, fontweight="bold", color="#111", zorder=4)
    if subtitle:
        ax.text(cx, y + h * 0.40, subtitle, ha="center", va="center",
                fontsize=8.2, color="#333", zorder=4)
    if stage:
        ax.text(x + w - 0.6, y + h - 1.2, stage, ha="right", va="top",
                fontsize=7.2, color=col, fontweight="bold", zorder=4)
    return (cx, y + h / 2), (x, x + w, y, y + h)


def arrow(ax, p0, p1, label="", color="#444", style="-|>", lw=2.0, ls="-",
          rad=0.0, lab_dy=1.6, lab_fs=8.0, lab_color=None):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=2)
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + lab_dy, label, ha="center", va="bottom",
                fontsize=lab_fs, color=lab_color or color, style="italic", zorder=5)


fig, ax = plt.subplots(figsize=(14, 8.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, 97, "GD-4D — Grounded Dreaming on a Frozen 4D Backbone",
        ha="center", va="top", fontsize=15, fontweight="bold")
ax.text(50, 92.5, "goal-image-conditioned robot policy;  the dreamed goal is injected into a queryable "
        "4D reconstruction, and two gates (Dₜ, τₜ) modulate the policy",
        ha="center", va="top", fontsize=9.5, color="#444")

# ---- top row: main data flow -------------------------------------------------
YM, HM = 58, 20
# inputs
_, ob = box(ax, 1.5, 68, 15, 11, "Current obs  $o_t$", "RGB frame(s)", "input")
_, ib = box(ax, 1.5, 52, 15, 11, "Instruction", "language", "input")
# dreamer
drm, db = box(ax, 21, YM, 16, HM, "Goal Dreamer", "SuSIE-recipe:\nIP2P fine-tuned on LIBERO", "open", "M2")
# bridge
brg, bb = box(ax, 42, YM, 16, HM, "LoRA Bridge", "inject dream as\nqueryable tokens", "open", "M1 / S3")
# backbone
bkb, kb = box(ax, 63, 52, 18, HM + 10, "Frozen 4D Backbone", "OpenD4RT ❄\nqueryable correspondences:\nxyz · uv · visibility", "validated", "S0", fs_title=11)
# policy
pol, pb = box(ax, 86, YM, 12.5, HM, "Diffusion\nPolicy", "actions $a_t$", "todo", "S6")

# flow arrows
arrow(ax, (ob[1], 73.5), (21, 71), "", "#666")
arrow(ax, (ib[1], 57.5), (21, 65), "", "#666")
arrow(ax, (db[1], YM + HM/2), (42, YM + HM/2), "dream $D$", "#B8860B", lab_color="#8a6400")
arrow(ax, (bb[1], YM + HM/2), (63, YM + HM/2), "dream tokens", "#666")
arrow(ax, (kb[1], YM + HM/2), (86, YM + HM/2), "features", "#666")
arrow(ax, (pb[1], YM + HM/2), (99.4, YM + HM/2), "", "#333")
ax.text(99.6, YM + HM/2 + 2.4, "→ robot", ha="right", fontsize=8.5, color="#333")

# ---- bottom row: the two heads (read backbone, gate policy) -------------------
YH, HH = 20, 18
dt, dtb = box(ax, 49, YH, 18, HH, "Disagreement head  $D_t$",
              "learned critic on $(o_t,$ dream)\nDINOv3 appearance features", "resolved", "S4 · M4 ✓")
tau, taub = box(ax, 71.5, YH, 15, HH, "Plan-position  $τ_t$",
                "“where in the plan?”", "todo", "S5")

# Dt reads (o_t, dream) appearance feats (NOT backbone geometry); tau reads backbone
arrow(ax, (46, 58.5), (55, YH + HH), "$o_t$, dream\nDINOv3 feats", "#0072B2", rad=0.18, lab_dy=0.4, lab_fs=7.2)
arrow(ax, (74, 52), (79, YH + HH), "reads", "#009E73", rad=0.15, lab_dy=0.2, lab_fs=7.5)
# heads -> policy (gate, dashed)
arrow(ax, (dtb[1], YH + HH - 2), (88, YM - 0.3), "gate", "#0072B2", ls="--", rad=-0.25, lab_dy=0.2, lab_fs=7.8)
arrow(ax, (taub[1], YH + HH), (89, YM - 0.3), "gate", "#777", ls="--", rad=-0.15, lab_dy=0.2, lab_fs=7.8)

# closed loop (re-dream) — arcs cleanly ABOVE the top row
arrow(ax, (88, 80), (31, 80), "closed loop: re-dream / re-plan  (S7)",
      "#888", ls=":", rad=0.22, lab_dy=3.6, lab_fs=8, lab_color="#666")

# ---- callout: Dt resolved (geometric route falsified -> learned critic) -------
co = FancyBboxPatch((1.5, 1.5), 97, 12.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                    linewidth=2.2, edgecolor=C["resolved"], facecolor=tint(C["resolved"], 0.09), zorder=1)
ax.add_patch(co)
ax.text(3.5, 11.9, "✓  $D_t$ resolved — the geometric read-out is falsified; $D_t$ is a LEARNED CRITIC (de-risked viable, M4)",
        ha="left", va="top", fontsize=10.5, fontweight="bold", color="#0a4a72")
ax.text(3.5, 8.4,
        "M1/M2/M3 falsified the frozen-backbone GEOMETRIC signal (self/cross-consistency at chance on real dreams:\n"
        "M2 delta 0.51, M3 ~0.60 on a plausible-but-wrong future).  M4: a learned critic on DINOv3 appearance "
        "features of $(o_t,$ dream)\nseparates correct vs wrong goals at AUROC 0.83–0.98 and transfers to generated "
        "dreams (0.78).   ⇒ $D_t$ = the plan’s S4-supervised\nappearance critic, NOT a backbone read-out; the frozen "
        "backbone’s role narrows to dream-conditioning.   Open: within-scene wrongness (in progress).",
        ha="left", va="top", fontsize=8.6, color="#123a52")

# ---- status legend -----------------------------------------------------------
lx, ly = 1.5, 87.0
items = [("validated", "validated (S0)"), ("open", "built · question open (M1/M2)"),
         ("resolved", "de-risked → build (S4·Dₜ, M4)"), ("todo", "not started (S5/S6/S7)")]
for i, (k, lab) in enumerate(items):
    x = lx + i * 24.5
    ax.add_patch(FancyBboxPatch((x, ly), 2.2, 2.2, boxstyle="round,pad=0.1,rounding_size=0.5",
                                linewidth=2, edgecolor=C[k], facecolor=tint(C[k], 0.3), zorder=4))
    ax.text(x + 3.0, ly + 1.1, lab, ha="left", va="center", fontsize=8.3, color="#222")

fig.tight_layout()
out = Path(__file__).resolve().parent / "fig0_pipeline_overview"
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print(f"[save] {out}.png / .svg / .pdf")

# Figures

## `fig0_pipeline_overview` — full GD-4D pipeline + de-risk status board

One self-explanatory figure of the whole pipeline: `obs + instruction → Goal Dreamer →
LoRA Bridge → Frozen 4D Backbone → Diffusion Policy → robot`, with the two gates
(`Dₜ` disagreement, `τₜ` plan-position) reading the backbone and gating the policy, plus
the closed loop (re-dream / re-plan).

Every stage is color-coded by de-risk status so the figure doubles as a project status board:
- **green** — validated (backbone, S0)
- **amber** — built, question still open (Goal Dreamer M2; LoRA Bridge M1/S3)
- **vermillion** — **falsified / blocked** (the disagreement head `Dₜ`, S4)
- **gray** — not started (`τₜ` S5, policy S6, executor S7)

The bottom callout records the current crossroads: `Dₜ` is falsified as a *geometric* read-out of
the frozen backbone by three converging de-risks (M1 premise, M2 real dreamer, M3 cross-consistency);
it must be **learned** (a critic) or sourced elsewhere (dreamer / policy uncertainty).

Rendered with a hand-coded matplotlib schematic (scipilot-figure-skill is a data-plot advisor and does
not do schematics — only its `setup_style` publication fonts are borrowed).

### Reproduce (env `gd4d5090`)
```bash
cd /workspace/code/GD_4D
/workspace/miniconda3/envs/gd4d5090/bin/python figures/pipeline_overview.py
# -> figures/fig0_pipeline_overview.{png,svg,pdf}
```

### Visual outputs (per project rule)
| file | produced by |
|---|---|
| `fig0_pipeline_overview.{png,svg,pdf}` | `pipeline_overview.py` |

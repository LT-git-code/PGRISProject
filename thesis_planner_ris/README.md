# Thesis main line: Planner-assisted / location-aware RIS control

**Goal.** Control RIS reflection phases using the robot’s **motion-planner
planned pose** so that cascaded AP–RIS–robot reflections stay pointed at the
robot as it moves. The default method is **planner-assisted / location-aware
RIS control** — not ML trajectory prediction.

**No ML in the main method.** Planned trajectories are a **synthetic stand-in**
for a real motion planner (e.g. MoveIt / nav stack). The default pipeline does
**not** import `phase2_traj`, load `.pth` checkpoints, or use torch.

---

## Design vs evaluation (hard rule)

| Quantity | Pose | Role |
|----------|------|------|
| Phase design | `p_plan` | Build `h_eff(p_plan)`, set `phi` |
| SNR evaluation | `p_true` | Always use `h_eff(p_true)` |

```
planner_pose_error_m = ||p_true - p_plan||_2
```

---

## Physics (reused from `phase1_mvp`)

Cascaded LoS channel (element-wise):

```
h_eff = h_ar ? h_ru
```

Unit-modulus phases maximizing coherent gain `phi^H h_eff`
(`np.vdot(phi, h_eff)`):

```
phi_n = h_eff,n / |h_eff,n| ,   |phi_n| = 1
```

SNR (RIS mode):

```
SNR = Pt * |phi^H h_eff(p_true)|^2 / sigma^2
```

### Modes

| Label | `phi` from | SNR on |
|-------|------------|--------|
| `no_ris` | — (direct path only) | `p_true` |
| `ideal_ris` | `h_eff(p_true)` | `p_true` |
| `planner_ris` | `h_eff(p_plan)` | `p_true` |

---

## Dependencies (default path)

- Required: `numpy`, `matplotlib`
- Forbidden in default path: `torch`, `phase2_traj`, any `.pth`
- Optional robust (Stage 3): wraps `phase3_robust` with planner uncertainty radius `R`

---

## How to run

From the repo root (`PGFinalProject/`):

```bash
# Stage 1 — baseline with p_true = p_plan (zero error)
python -m thesis_planner_ris.run_stage1

# Stage 2 — sweep planner_pose_error_m (0 .. ~2 m)
python -m thesis_planner_ris.run_stage2
# -> outputs/snr_vs_planner_error.csv
# -> outputs/fig_snr_vs_planner_error.png

# Stage 3 — optional robust (skipped; prints future-work note)
python -m thesis_planner_ris.run_stage3

# Stage 4 — demo (planner labels; no ML)
python -m thesis_planner_ris.demo.run_demo --gif
# -> outputs/demo_planner_ris.gif
# -> outputs/demo_preview.png
```

Artifacts land under `thesis_planner_ris/outputs/`.

**Stage 3 note:** Robust wrapping of `phase3_robust` is skipped; documented as future work (planner uncertainty radius `R`, not LSTM ADE).

---

## Legacy folder map

| Legacy | Role for this main line |
|--------|-------------------------|
| `phase1_mvp` | **Imported** — geometry / channel / beamforming / snr |
| `phase2_traj` | **Ignored** by default (early ML exploration; appendix only) |
| `phase3_robust` | Optional wrapper — narrative: planner uncertainty `R` |
| `phase4_demo` | Viz reference; new demo must not call ML |

---

## Layout

```
thesis_planner_ris/
  README.md, config.py, planner_traj.py, modes.py
  experiments/   # Stage 2+
  demo/          # Stage 4
  robust/        # Stage 3 optional
  thesis/        # Stage 5 packaging
  outputs/
  run_stage1.py, run_stage2.py, run_stage3.py
```

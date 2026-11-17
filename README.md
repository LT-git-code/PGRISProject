# PGRISProject

**Planner-assisted / location-aware RIS control** for a moving robot.

Use the motion planner’s **planned pose** to set RIS phases so the cascaded AP–RIS–robot link stays aligned while the robot moves. The default method is **not** ML trajectory prediction.

```
PGRISProject/
├── Physical_model/       # Geometry, channel, beamforming, SNR, traj, modes
├── Experiment_script/    # Stage 1–4 / MVP runners, sweeps, demo
└── output/               # All figures, CSV, summaries, demo media
```

---

## 1. Idea

| Symbol | Meaning |
|--------|---------|
| `p_plan` | Planned pose from the motion planner (known to RIS) |
| `p_true` | True / executed pose (used only for evaluation) |
| `planner_pose_error_m` | \(\|p_{\mathrm{true}}-p_{\mathrm{plan}}\|_2\) |

**Hard rule — design vs evaluation**

| Step | Pose used | What happens |
|------|-----------|--------------|
| Phase design | `p_plan` | Build \(h_{\mathrm{eff}}(p_{\mathrm{plan}})\), set \(\varphi\) |
| SNR evaluation | `p_true` | Always evaluate on \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) or the direct link |

`p_plan` is a **synthetic arc** standing in for MoveIt / a nav stack. No torch / `.pth` on the default path.

---

## 2. Method

### Modes

| Mode | \(\varphi\) from | SNR evaluated on | Role |
|------|------------------|------------------|------|
| `no_ris` | — (direct AP→robot) | `p_true` | Lower baseline |
| `ideal_ris` | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) | Oracle upper bound |
| `planner_ris` | \(h_{\mathrm{eff}}(p_{\mathrm{plan}})\) | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) | **Proposed method** |

### Physics (`Physical_model`)

- LoS free-space gain; cascade \(h_{\mathrm{eff}} = h_{\mathrm{ar}} \odot h_{\mathrm{ru}}\)
- Unit-modulus phase alignment; SNR on cascade (RIS) or blocked/open direct link (`BLOCK_DIRECT_LINK`)

### Pipeline

```
Physical_model.planner_traj  →  p_plan, p_true
        → Physical_model.modes.simulate_modes
        → Experiment_script (plots / CSV / demo)
        → output/
```

---

## 3. How to run

From the repo root (`PGRISProject/`). Need: `numpy`, `matplotlib`.

```bash
# Stage 1 — baseline: p_true = p_plan
python -m Experiment_script.run_stage1

# Stage 2 — planner pose-error sweep
python -m Experiment_script.run_stage2

# Stage 3 — robust (skipped / future work)
python -m Experiment_script.run_stage3

# Stage 4 — demo
python -m Experiment_script.demo.run_demo --gif

# Optional physics MVP smoke test
python -m Experiment_script.run_mvp
```

All artifacts go to `output/`.

---

## 4. Layout

### `Physical_model/`

| File | Role |
|------|------|
| `config.py` | RF / geometry / stage & demo constants; `OUTPUTS_DIR → output/` |
| `geometry.py` | AP / RIS element positions |
| `channel.py` | LoS cascade + direct link |
| `beamforming.py` | Deterministic \(\varphi\) |
| `snr.py` | SNR utilities |
| `planner_traj.py` | Synthetic `p_plan` / `p_true` |
| `modes.py` | `no_ris` / `ideal_ris` / `planner_ris` |

### `Experiment_script/`

| File / package | Role |
|----------------|------|
| `run_stage1.py` / `run_stage2.py` / `run_stage3.py` | Thesis stage entries |
| `experiments/` | Error sweep, CSV, paper plots |
| `demo/` | Animation / GIF |
| `run_mvp.py` / `simulation.py` / `plot_snr.py` | Legacy Phase-1 MVP entry |
| `predictor.py` | Stub only (not used by main line) |

### `output/`

Unified destination for PNG / GIF / CSV / summary text.

---

## 5. Status

| Item | Status |
|------|--------|
| Stage 1 / 2 / 4 | Implemented |
| Stage 3 robust | Skipped — future work |
| Spatial NLoS scenario | Planned (direct AP–robot NLoS; RIS cascade LoS) |
| Real motion planner | Not connected (synthetic arc) |

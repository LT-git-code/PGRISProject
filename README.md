# PGRISProject

**Planner-assisted / location-aware RIS control** for a moving robot.

Use the motion planner’s **planned pose** to set RIS phases so the cascaded AP–RIS–robot link stays aligned while the robot moves. The main method is **not** ML trajectory prediction.

```
PGRISProject/
├── phase1_mvp/            # Physics: geometry, channel, phase, SNR
└── thesis_planner_ris/    # Main line: p_plan/p_true, modes, Stage 1–4
```

---

## 1. Idea

A robot is commanded along a planned path. The RIS controller usually does **not** know the exact executed pose in real time, but it **does** know the planner output.

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

In this repo, `p_plan` is a **synthetic arc** standing in for MoveIt / a nav stack. No torch, no `.pth`, no `phase2_traj` on the default path.

---

## 2. Method

### 2.1 Three comparison modes

| Mode | \(\varphi\) from | SNR evaluated on | Role |
|------|------------------|------------------|------|
| `no_ris` | — (direct AP→robot) | `p_true` | Lower baseline |
| `ideal_ris` | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) | Oracle upper bound |
| `planner_ris` | \(h_{\mathrm{eff}}(p_{\mathrm{plan}})\) | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) | **Proposed method** |

Expected behaviour:

- Zero error (`p_true = p_plan`): `ideal_ris` ≈ `planner_ris` ≫ `no_ris`
- Growing pose error: `planner_ris` falls away from `ideal_ris`

### 2.2 Physics (implemented in `phase1_mvp`)

- LoS free-space gain: Friis amplitude \(\lambda/(4\pi d)\), phase \(e^{-j 2\pi d/\lambda}\)
- Cascaded channel (element-wise): \(h_{\mathrm{eff}} = h_{\mathrm{ar}} \odot h_{\mathrm{ru}}\)
- Unit-modulus phase: \(\varphi_n = h_{\mathrm{eff},n}/|h_{\mathrm{eff},n}|\), gain \(\varphi^H h_{\mathrm{eff}} = \|h_{\mathrm{eff}}\|_1\) (`np.vdot`)
- RIS SNR: \(\mathrm{SNR} = P_t\,|\varphi^H h_{\mathrm{eff}}|^2 / \sigma^2\) (cascade only; no direct term mixed in)
- No-RIS: blocked / open direct link via `BLOCK_DIRECT_LINK`

### 2.3 Pipeline

```
generate p_plan  →  execute with noise → p_true
                         │
                         ▼
              modes.simulate_modes
                 ├─ no_ris(p_true)
                 ├─ ideal_ris(p_true)
                 └─ planner_ris(p_plan → φ, p_true → SNR)
                         │
                         ▼
              plots / CSV / demo / summaries
```

Core code path:

1. `thesis_planner_ris/planner_traj.py` — synthetic plan + execution noise  
2. `thesis_planner_ris/modes.py` — three modes (imports phase-1 physics)  
3. `phase1_mvp/{geometry,channel,beamforming,snr}.py` — RF model  

---

## 3. Experiments

All thesis stages run from the **repo root**. Need: `numpy`, `matplotlib`.

### Stage 1 — Sanity baseline (zero error)

**Question.** If tracking is perfect, does planner-based RIS match the oracle?

```bash
python -m thesis_planner_ris.run_stage1
```

| Setting | Value |
|---------|--------|
| Execution noise | \(\sigma = 0\) → `p_true = p_plan` |
| Check | `Ideal ≈ Planner` and both ≫ `NoRIS` |
| Outputs | `outputs/snr_planner_baseline.png`, `stage1_summary.txt` |

### Stage 2 — Sensitivity to planner pose error

**Question.** How does `planner_ris` degrade as execution / tracking error grows?

```bash
python -m thesis_planner_ris.run_stage2
```

| Setting | Value |
|---------|--------|
| Noise std sweep | `0, 0.1, 0.3, 0.5, 1.0, 2.0` m |
| Monte Carlo | 8 trials per std |
| X-axis | empirical mean `planner_pose_error_m` (not ML ADE) |
| Check | match at zero error; `planner_ris` tends to drop as error grows |
| Outputs | `snr_vs_planner_error.csv`, `fig_snr_vs_planner_error.png`, `stage2_summary.txt` |

### Stage 3 — Robust RIS (optional, skipped)

```bash
python -m thesis_planner_ris.run_stage3
```

Documented future work: design \(\varphi\) under a planner-uncertainty ball of radius \(R\). Not required for the current main line.

### Stage 4 — Visual demo

```bash
python -m thesis_planner_ris.demo.run_demo --gif
```

Shows `p_plan` vs `p_true` and illustrative beams (schematic only, not a full EM pattern).  
Outputs: `demo_preview.png`, `demo_planner_ris.gif`.

### Physics smoke test (optional)

```bash
python -m phase1_mvp.run_mvp
```

Standalone Phase-1 SNR comparison plot under `phase1_mvp/outputs/`.

---

## 4. Code layout

### `phase1_mvp/` — physics layer

| File | Role |
|------|------|
| `config.py` | Carrier, power, RIS array, AP/RIS poses, direct-link switch |
| `geometry.py` | AP position; RIS X–Z URA element positions |
| `channel.py` | Free-space LoS, cascade \(h_{\mathrm{eff}}\), direct link |
| `beamforming.py` | Deterministic unit-modulus \(\varphi\) |
| `snr.py` | dBm→watt, SNR in dB |
| `run_mvp.py` | Standalone MVP entry |
| `simulation.py` / `plot_snr.py` | Legacy MVP curves |
| `predictor.py` | Stub only — **not** used by the thesis main line |

### `thesis_planner_ris/` — thesis main line

```
thesis_planner_ris/
├── config.py              # Re-exports phase-1 RF/geometry; stage / demo knobs
├── planner_traj.py        # Synthetic p_plan; p_true = p_plan + noise
├── modes.py               # no_ris / ideal_ris / planner_ris
├── run_stage1.py
├── run_stage2.py
├── run_stage3.py          # Prints skip / future-work note
├── experiments/           # Error sweep, CSV, paper plots
├── demo/                  # Animation pipeline + GIF export
├── robust/                # Placeholder for Stage 3
├── thesis/                # Packaging placeholder
└── outputs/               # Figures, CSV, summaries, demo media
```

| File / package | Role |
|----------------|------|
| `planner_traj.py` | `generate_planner_trajectory`, `execute_with_error`, `make_paired_trajectories` |
| `modes.py` | `design_phi_from_plan`, `snr_*`, `simulate_modes` |
| `experiments/error_sweep.py` | Stage-2 Monte Carlo sweep + acceptance checks |
| `experiments/export_csv.py` | Write sweep table |
| `experiments/paper_plots.py` | SNR vs error figure |
| `demo/pipeline.py` | Demo loop over time |
| `demo/animate_mpl.py` / `export_video.py` | Preview + GIF |

---

## 5. Status

| Item | Status |
|------|--------|
| Stage 1 baseline | Done |
| Stage 2 error sweep | Done |
| Stage 4 demo | Done |
| Stage 3 robust (\(R\)-ball) | Skipped — future work |
| Real planner (MoveIt / nav) | Not connected (synthetic arc) |
| ML / LSTM prediction | Out of main line |

---

## 6. Reading order (for code walkthrough)

1. `thesis_planner_ris/planner_traj.py` — where `p_plan` / `p_true` come from  
2. `thesis_planner_ris/modes.py` — especially `snr_planner_ris`  
3. `phase1_mvp/channel.py` → `beamforming.py` → `snr.py`  
4. `run_stage1.py` → `run_stage2.py` → `demo/run_demo.py`  

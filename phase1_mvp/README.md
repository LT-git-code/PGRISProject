# Phase-1 MVP

RIS-aided AP–robot link SNR simulation (numpy / matplotlib only).

## Run

```bash
python -m phase1_mvp.run_mvp
```

Output: `phase1_mvp/outputs/snr_comparison.png`.

## Review notes (phase-1)

- Conventions are unified on **column vectors** and **Hermitian** coherent gain \(\varphi^H h_{\mathrm{eff}}\) (`np.vdot(phi, h_eff)`).
- Cascaded channel is **element-wise** \(h_{\mathrm{eff}} = h_{\mathrm{ar}} \odot h_{\mathrm{ru}}\) (not a matrix product).
- RIS SNR curves use the **cascade only** (no direct term added into Ideal/Noisy).
- No-RIS uses the scalar direct link with switch `BLOCK_DIRECT_LINK`.
- Phase-1 “noisy” pose is a stand-in; phase-2 swaps it for `trajectory_predictor` output (see below). **Do not implement LSTM here.**

## Signal model (canonical)

Free-space path gain (distance \(d\), wavelength \(\lambda\)):

\[
h_{\mathrm{FS}}(d)
  = \frac{\lambda}{4\pi d}\,
    e^{-j 2\pi d / \lambda}.
\]

AP\(\to\)RIS and RIS\(\to\)robot vectors (\(N\) elements):

\[
[h_{\mathrm{ar}}]_n = h_{\mathrm{FS}}(\|p_{\mathrm{RIS},n}-p_{\mathrm{AP}}\|),
\quad
[h_{\mathrm{ru}}]_n = h_{\mathrm{FS}}(\|p_{\mathrm{RIS},n}-p_{\mathrm{U}}\|),
\quad
h_{\mathrm{eff}} = h_{\mathrm{ar}} \odot h_{\mathrm{ru}}.
\]

Unit-modulus phases maximizing \(|\varphi^H h_{\mathrm{eff}}|\):

\[
\varphi_n
  = \frac{[h_{\mathrm{eff}}]_n}{|[h_{\mathrm{eff}}]_n|}
  = e^{+j\angle [h_{\mathrm{eff}}]_n}
  \quad(|\varphi_n|=1),
\qquad
\varphi^H h_{\mathrm{eff}}
  = \|h_{\mathrm{eff}}\|_1.
\]

Powers \(P_t,\sigma^2\) are **linear watts** (`dbm_to_watt`).

\[
\mathrm{SNR}_{\mathrm{RIS}}
  = \frac{P_t\,|\varphi^H h_{\mathrm{eff}}|^2}{\sigma^2},
\qquad
\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}(\mathrm{SNR}).
\]

Direct AP\(\to\)robot (No-RIS baseline), with blocking switch:

\[
h_{\mathrm{direct}}
  = \alpha\, h_{\mathrm{FS}}(\|p_{\mathrm{U}}-p_{\mathrm{AP}}\|),
\quad
\alpha =
\begin{cases}
\alpha_{\mathrm{blk}} & \text{if } \texttt{block\_direct\_link}=\texttt{True},\\
\alpha_{\mathrm{open}} & \text{otherwise (default }1\text{)}.
\end{cases}
\]

\[
\mathrm{SNR}_{\mathrm{no\text{-}RIS}}
  = \frac{P_t\,|h_{\mathrm{direct}}|^2}{\sigma^2}.
\]

Config: `BLOCK_DIRECT_LINK`, `DIRECT_PATH_ATTENUATION_BLOCKED`, `DIRECT_PATH_ATTENUATION_OPEN`.

## Phase-1 curves vs design pose

| Curve | \(\varphi\) from | SNR on |
|-------|------------------|--------|
| No RIS | — | \(h_{\mathrm{direct}}(p_{\mathrm{true}})\) |
| Ideal RIS | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) |
| Noisy-position RIS | \(h_{\mathrm{eff}}(p_{\mathrm{design}})\) | \(h_{\mathrm{eff}}(p_{\mathrm{true}})\) |

Phase-1: \(p_{\mathrm{design}} = p_{\mathrm{true}} + \mathcal{N}(0,\sigma^2 I)\).  
Phase-2: \(p_{\mathrm{design}} \leftarrow\) predicted pose (below).

## Phase-2 interface draft

```python
# phase1_mvp/predictor.py
def trajectory_predictor(history: np.ndarray) -> np.ndarray:
    """
    history: (T_hist, 3)  # meters, oldest → newest
    returns: (T_pred, 3)  # meters, future positions
    """
```

- `T_hist = config.T_HIST`, `T_PRED = config.T_PRED` (sizes only; no LSTM yet).
- Wire-in point: `simulation.simulate_snr_curves` — replace Gaussian `p_design` with `trajectory_predictor(history)[k]`; keep evaluation on `p_true`.

## Layout

| Module | Role |
|--------|------|
| `config.py` | Constants, `BLOCK_DIRECT_LINK`, predictor window sizes |
| `geometry.py` | AP / RIS element positions |
| `channel.py` | LoS, cascade, direct (+ blocking) |
| `beamforming.py` | Deterministic \(\varphi\) |
| `snr.py` | Watt conversion & SNR |
| `simulation.py` | Trajectory + SNR curves |
| `predictor.py` | Phase-2 predictor stub |
| `plot_snr.py` | SNR comparison plot |
| `run_mvp.py` | CLI entry |

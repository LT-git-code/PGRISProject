# Planner-Assisted RIS Control for a Moving Robot

This project simulates reconfigurable intelligent surface (RIS) assistance for a moving robot. A planned robot pose is used to configure RIS phases, and the resulting signal-to-noise ratio (SNR) is evaluated at the robot's actual pose. The current workflow is a physics-based simulation; it does not use machine-learning trajectory prediction.

## Current workflow

1. Generate a planned path `p_plan` using the configured trajectory source. The default is a built-in 2D occupancy-grid A* planner; the legacy synthetic `arc` path is also available.
2. Simulate execution uncertainty by adding Gaussian position noise to `p_plan`, producing `p_true`. With zero noise, the two paths are identical.
3. Compare three communication modes at each time step:
   - `no_ris`: direct AP-to-robot link only.
   - `ideal_ris`: configure RIS using the actual pose `p_true`. This is an oracle reference, not a deployable controller.
   - `planner_ris`: configure RIS using the planned pose `p_plan`, then evaluate SNR at `p_true`. This is the proposed planner-assisted mode.
4. Save plots, CSV data, summaries, and demo media under `output/`.

The channel model uses free-space path gain for the AP–RIS and RIS–robot links, a cascaded RIS channel, and deterministic phase alignment. The direct link can be configured as blocked or open.

## Run the experiments

Run commands from the repository root. The scripts require Python, NumPy, and Matplotlib. GIF export additionally uses Pillow.

```bash
# Generate zero-error and execution-error SNR time-series plots
python -m Experiment_script.run_planner_ris_snr_baselines

# Sweep execution error and export the SNR comparison
python -m Experiment_script.run_planner_ris_snr_error_sweep

# Generate a static demo preview
python -m Experiment_script.demo.run_demo --preview-only

# Generate a static preview and animated GIF
python -m Experiment_script.demo.run_demo --gif
```

### Baseline plots

The baseline runner creates `output/snr_planner_baseline.png` for zero execution error and `output/snr_error_baseline.png` for nonzero execution error. Both plots use Time step on the horizontal axis and SNR (dB) on the vertical axis. The execution-error plot uses the configured `ERROR_BASELINE_STD_M` value (default 0.5 m per coordinate axis). A summary is written to `output/stage1_summary.txt`.

### Error sweep

The sweep tests execution-noise standard deviations of 0, 0.1, 0.3, 0.5, 1.0, and 2.0 m per coordinate axis. Each setting uses 8 Monte Carlo trials and 40 time steps by default. It writes:

- `output/snr_vs_planner_error.csv`
- `output/fig_snr_vs_planner_error.png`
- `output/stage2_summary.txt`

In the sweep figure, the horizontal axis is the empirical mean distance between `p_plan` and `p_true`, in meters. It is not the input noise standard deviation or the time step. The vertical axis is mean SNR in dB.

### Demo options

The demo defaults to 120 time steps and an execution-noise standard deviation of 0.5 m per coordinate axis. Use `--steps` and `--error-std` to override these defaults; `--show` opens an interactive window. The static preview is saved as `output/demo_preview.png`; `--gif` also saves `output/demo_planner_ris.gif`.

## Main configuration

Simulation and experiment parameters are centralized in `Physical_model/config.py`. Current defaults include:

- Carrier frequency: 5 GHz; transmit power: 30 dBm; noise power: -90 dBm.
- RIS array: 8 by 8 elements, with half-wavelength spacing.
- Direct AP-to-robot link: blocked by default.
- Grid planner map: x from 0 to 100 m and y from -5 to 45 m; resolution 0.5 m. The configured start is (15, 34) m and the goal is (85, 34) m. Two rectangular obstacles and a 0.8 m robot radius are used for planning.
- Baseline, sweep, and demo trajectory lengths: 100, 40, and 120 time steps respectively.

Change the centralized configuration to run a different scenario. The generated artifacts reflect the parameters used when each experiment was run.

## Project layout

- `Physical_model/`: geometry, channel, beamforming, SNR, trajectory generation, and the grid motion planner.
- `Experiment_script/run_planner_ris_snr_baselines.py`: zero-error and execution-error baseline plots.
- `Experiment_script/run_planner_ris_snr_error_sweep.py`: Monte Carlo error sweep, CSV export, figure, and summary.
- `Experiment_script/experiments/`: sweep, export, and plotting components.
- `Experiment_script/demo/`: demo frame generation, preview, and GIF export.
- `output/`: generated figures, CSV files, summaries, and demo media.

## Scope and limitations

The built-in A* planner generates a simulated path; it is not connected to a physical robot or an external motion-planning system. Execution error is represented by random position noise. The `ideal_ris` mode assumes access to the actual pose and is included only as an upper-bound reference. The current model does not include measured channel data, hardware phase quantization, or a real-world validation campaign. A robust-control Stage 3 runner is not currently included.

---

# 中文翻译

## 面向移动机器人的规划辅助 RIS 控制

本项目仿真可重构智能表面（RIS）对移动机器人的通信辅助效果：根据机器人的计划位姿配置 RIS 相位，并在实际位姿处评估信噪比（SNR）。当前流程是基于物理模型的仿真，不使用机器学习进行轨迹预测。

## 当前流程

1. 根据配置的轨迹来源生成计划路径 `p_plan`。默认使用内置的二维占用栅格 A* 规划器，也可选择旧版合成圆弧路径 `arc`。
2. 在 `p_plan` 上叠加高斯位置噪声，模拟执行误差并得到 `p_true`。误差为零时，两条轨迹重合。
3. 在每个时间步比较三种通信模式：
   - `no_ris`：仅使用 AP 到机器人的直连链路。
   - `ideal_ris`：按实际位姿 `p_true` 配置 RIS，作为理想参照；它不是可部署的控制器。
   - `planner_ris`：按计划位姿 `p_plan` 配置 RIS，再在 `p_true` 处评估 SNR；这是规划辅助方案。
4. 将图表、CSV 数据、摘要和演示媒体保存到 `output/`。

信道模型使用 AP–RIS 和 RIS–机器人链路的自由空间路径增益、RIS 级联信道及确定性相位对齐。直连链路可以配置为阻挡或开放。

## 运行实验

请从仓库根目录运行命令。脚本需要 Python、NumPy 和 Matplotlib；导出 GIF 还需要 Pillow。

```bash
# 生成零误差和执行误差下的 SNR 时间序列图
python -m Experiment_script.run_planner_ris_snr_baselines

# 扫描执行误差并导出 SNR 对比结果
python -m Experiment_script.run_planner_ris_snr_error_sweep

# 仅生成静态演示预览图
python -m Experiment_script.demo.run_demo --preview-only

# 生成静态预览图和动态 GIF
python -m Experiment_script.demo.run_demo --gif
```

### 基线图

基线脚本生成 `output/snr_planner_baseline.png`（零执行误差）和 `output/snr_error_baseline.png`（非零执行误差）。两张图的横轴均为 Time step，纵轴均为 SNR（dB）。执行误差图使用配置项 `ERROR_BASELINE_STD_M`，默认值为每个坐标轴 0.5 m。摘要写入 `output/stage1_summary.txt`。

### 误差扫描

扫描每个坐标轴上的执行噪声标准差：0、0.1、0.3、0.5、1.0 和 2.0 m。每个设置默认进行 8 次蒙特卡洛试验，每次包含 40 个时间步。输出文件为：

- `output/snr_vs_planner_error.csv`
- `output/fig_snr_vs_planner_error.png`
- `output/stage2_summary.txt`

扫描图的横轴是 `p_plan` 与 `p_true` 之间的实测平均距离，单位为米；它不是输入噪声标准差，也不是时间步。纵轴是平均 SNR，单位为 dB。

### 演示选项

演示默认使用 120 个时间步，每个坐标轴的执行噪声标准差为 0.5 m。可以使用 `--steps` 和 `--error-std` 修改默认值；`--show` 会打开交互窗口。静态预览保存为 `output/demo_preview.png`；使用 `--gif` 时还会生成 `output/demo_planner_ris.gif`。

## 主要配置

仿真和实验参数集中在 `Physical_model/config.py`。当前默认值包括：

- 载波频率 5 GHz；发射功率 30 dBm；噪声功率 -90 dBm。
- RIS 阵列为 8×8 个单元，单元间距为半波长。
- 默认阻挡 AP 到机器人的直连链路。
- 栅格规划地图：x 范围 0–100 m，y 范围 -5–45 m，分辨率 0.5 m。起点为 (15, 34) m，终点为 (85, 34) m。路径规划使用两个矩形障碍物，并考虑 0.8 m 的机器人半径。
- 基线、误差扫描和演示的轨迹长度分别为 100、40 和 120 个时间步。

修改集中配置即可运行不同场景。生成的文件对应各次实验运行时所使用的参数。

## 项目结构

- `Physical_model/`：几何、信道、波束成形、SNR、轨迹生成和栅格运动规划。
- `Experiment_script/run_planner_ris_snr_baselines.py`：零误差和执行误差基线图。
- `Experiment_script/run_planner_ris_snr_error_sweep.py`：蒙特卡洛误差扫描、CSV 导出、绘图和摘要。
- `Experiment_script/experiments/`：扫描、导出和绘图组件。
- `Experiment_script/demo/`：演示帧生成、预览图和 GIF 导出。
- `output/`：生成的图表、CSV、摘要和演示媒体。

## 适用范围与局限

内置 A* 规划器生成的是仿真路径，尚未连接实体机器人或外部运动规划系统。执行误差通过随机位置噪声建模。`ideal_ris` 假设可以获得实际位姿，仅作为上限参照。当前模型不包含实测信道数据、硬件相位量化或现实场景验证。项目目前没有实现鲁棒控制 Stage 3 入口。

## Figure 3

S3 calibration curves and fitted piecewise model used by the planner.

这个目录把 Figure 3 整理成独立实验目录，保留三层内容：

1. `run_data.py`
   冻结当前论文使用的校准锚点和 dense probe 数据，或重新调用 S3 probe 脚本。
2. `generate.py`
   从本目录 CSV 重新拟合 2 段 piecewise 模型，并生成 Figure 3。
3. 本地数据文件
   - `data/anchor_points.csv`
   - `data/dense_probe.csv`

### 当前设计

- 主图使用 full-range 的稀疏 calibration anchors，覆盖 `1 KiB` 到 `256 MiB`
- inset 使用 dense serial probe，覆盖 `0.5 MiB` 到 `16 MiB`
- 拟合时同时使用两类观测，但对 dense probe 区间赋更高权重，以降低 Figure 3 里最明显的偏差

### 如何冻结当前论文数据

```bash
python3 runners/s3-piecewise-calibration-curves/run_data.py
```

This offline path copies the frozen paper probe into `data/latest/`. To convert
an existing raw probe, pass `--probe-stats-json path/to/serial_stats.json`.

### 如何重新做一次 dense probe

```bash
python3 runners/s3-piecewise-calibration-curves/run_data.py --rerun-probe
```

The live probe reads `VECIO_S3_BUCKET` and `VECIO_S3_REGION`, matching
`scripts/run_all.sh`. For a one-off run, pass `--s3-bucket` and
`--s3-region` explicitly. `--config <yaml>` remains available only for
archived protocols that carry their own endpoint file.

`--rerun-probe` 会调用：

- `runners/s3-piecewise-calibration-curves/probe_s3_range_cost_model.py`

并使用当前论文那套 dense probe 区间：

- `0.5 MiB` 到 `16 MiB`
- 步长 `0.5 MiB`
- serial-only

### 如何重画当前图片

```bash
python3 runners/s3-piecewise-calibration-curves/run.py
```

### 输出文件

- `figure_03_s3_calibration_curves_piecewise_model.png`
- `figure_03_s3_calibration_curves_piecewise_model.pdf`
- `fit_summary.json`

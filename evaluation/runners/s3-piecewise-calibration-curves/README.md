# Figure 4: S3 latency calibration

Prepares serial Range-GET samples and the fitted latency model for Figure 4.
Run from `evaluation/`.

## Prepare data

Copy the frozen paper samples into the fresh-output directory:

```bash
python3 -m artifact run s3-piecewise-calibration-curves
```

To convert a recorded probe instead, pass
`--probe-stats-json path/to/serial_stats.json` to `run_data.py`.

## Collect new samples

```bash
python3 runners/s3-piecewise-calibration-curves/run_data.py --rerun-probe
```

Set `VECIO_S3_BUCKET` and `VECIO_S3_REGION`, or pass `--s3-bucket` and
`--s3-region`. The live probe uses serial reads from 0.5 to 16 MiB in 0.5 MiB
steps. `--config <yaml>` supports older protocols with an endpoint file.

## Render

```bash
python3 -m artifact render figure-04-object-store-io
```

Stage outputs are stored under
`experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves/data/latest/`.
They include `dense_probe.csv`, `source_latency_model.json`, calibration
summaries, and provenance. Figure rendering belongs to the parent experiment.

# Background: S3 piecewise calibration curves

Feeds the parent experiment.

Piecewise calibration of S3 GET latency used by the background microbenchmark figure.

Spec: `s3-piecewise-calibration-curves`. The parent experiment provides the paper figure.

## Directory layout

| Path | Contents |
| --- | --- |
| `data/paper/` | Frozen inputs |
| `data/latest/` | New stage results |

The parent experiment renders the paper figure from these stage results.

## What this experiment produces

- `data/paper/dense_probe.csv`
- `data/paper/source_latency_model.json`

## Backend requirements

- Requires live S3 access via AWS credentials + bucket.

## Running

Run from this directory:

```bash
# Refresh this measurement stage:
./run.sh

# Render the parent figure from frozen measurements:
../../render.sh light
```

The wrappers call the commands in `artifact/registry.py`.

# Intro: S3 range-GET latency decomposition

Renders the recorded S3 Range-GET latency decomposition. The run script
exports eight fixed measurement rows; it does not collect new S3 measurements.

Spec: `figure-01-s3-range-get-latency`. Paper labels: fig:intro-latency-composition.

## Directory layout

| Path | Contents |
| --- | --- |
| `data/paper/` | Frozen measurements |
| `data/latest/` | CSV exported from the fixed measurement rows |
| `_regen/` | Outputs rendered from frozen measurements |
| `_latest/` | Outputs rendered from the exported CSV |

Committed figures and tables are the paper outputs. Fresh renders
go to the directories above.

## What this experiment produces

- `experiments/figure-01-s3-range-get-latency/latency_composition.pdf`

## Backend requirements

- Runs locally; no external services.

## Running

Run from this directory:

```bash
# Export the fixed measurement rows and render them
./run.sh
./render.sh

# Render frozen measurements
./render.sh light
```

The wrappers call the commands in `artifact/registry.py`.

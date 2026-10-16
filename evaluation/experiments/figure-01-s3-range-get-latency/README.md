# Intro: S3 range-GET latency decomposition

Latency decomposition microbenchmark illustrating per-phase contributions of an S3 range-GET.

Spec: `figure-01-s3-range-get-latency`. Paper labels: fig:intro-latency-composition.

## Directory layout

| Path | Contents |
| --- | --- |
| `data/paper/` | Frozen measurements |
| `data/latest/` | New measurements |
| `_regen/` | Outputs rendered from frozen measurements |
| `_latest/` | Outputs rendered from new measurements |

Committed figures and tables are the paper outputs. Fresh renders
go to the directories above.

## What this experiment produces

- `experiments/figure-01-s3-range-get-latency/latency_composition.pdf`

## Backend requirements

- Runs locally; no external services.

## Running

Run from this directory:

```bash
# Collect measurements and render the result
./run.sh
./render.sh

# Render frozen measurements
./render.sh light
```

The wrappers call the commands in `artifact/registry.py`.

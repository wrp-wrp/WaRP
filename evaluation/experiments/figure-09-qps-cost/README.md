# Evaluation: all-6-dataset QPS-vs-cost grid

Compares throughput and request reduction across six datasets at `Q=4`, `IO=64`.

Spec: `figure-09-qps-cost`. Paper labels: fig:eval-qps-cost.

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

- `experiments/figure-09-qps-cost/figure_qps_vs_savings_q4_io64_all6_grid_qps_p95_3x4.pdf`

## Backend requirements

- Requires a built Lance example binary (see `scripts/setup_ec2.sh`).
- Requires live S3 access via AWS credentials + bucket.

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

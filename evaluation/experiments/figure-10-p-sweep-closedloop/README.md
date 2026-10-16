# Evaluation: closed-loop p-sweep QPS/latency tradeoff

Compares throughput and latency while varying the planner parallelism estimate `P`.

Spec: `figure-10-p-sweep-closedloop`. Paper labels: fig:eval-p-sweep-closedloop.

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

- `experiments/figure-10-p-sweep-closedloop/figure_closed_loop_env3_reorder_only_combined.pdf`

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

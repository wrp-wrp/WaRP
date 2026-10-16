# Evaluation: real-data planner overhead (figure + table)

Measures planner overhead and produces the four-way figure and its TeX table.

Spec: `figure-12-planner-overhead`. Paper labels: fig:eval-planner-overhead-breakdown, tab:eval-planner-overhead.

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

- `experiments/figure-12-planner-overhead/figure_03_real_data_planner_overhead_piecewise.pdf`
- `experiments/figure-12-planner-overhead/table_03_real_data_planner_overhead_piecewise.tex`

## Backend requirements

- Requires a built Lance example binary (see `scripts/setup_ec2.sh`).

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

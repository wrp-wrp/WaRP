# Stage: offline reordering cost sweep

Feeds the parent experiment.

Offline reordering runtime sweep across methods and cluster counts; feeds the figure-13-reordering-methods left panel.

Spec: `reorder-cost`. The parent experiment provides the paper figure.

## Directory layout

| Path | Contents |
| --- | --- |
| `data/paper/` | Frozen inputs |
| `data/latest/` | New stage results |

The parent experiment renders the paper figure from these stage results.

## What this experiment produces

- `experiments/figure-13-reordering-methods/stages/reorder-cost/data/paper/figure_reorder_cost_cluster_sweep.csv`

## Backend requirements

- Runs locally; no external services.

## Running

Run from this directory:

```bash
# Refresh this measurement stage:
./run.sh

# Render the parent figure from frozen measurements:
../../render.sh light
```

The wrappers call the commands in `artifact/registry.py`.

# Intro: fine vs coarse clustering recall-matched comparison

Compares fine and coarse IVF clustering at matched recall for Table 1.

Spec: `table-01-cluster-coarsening`. Paper labels: tab:intro-coarse-cluster.

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

- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.csv`
- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex`

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

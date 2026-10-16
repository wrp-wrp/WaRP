# Stage: locality-aware co-access gap

Feeds the parent experiment.

Locality-quality signal computed per reordering method; feeds the figure-13-reordering-methods right panel.

Spec: `locality-aware-coaccess-gap`. The parent experiment provides the paper figure.

## Directory layout

| Path | Contents |
| --- | --- |
| `data/paper/` | Frozen inputs |
| `data/latest/` | New stage results |

The parent experiment renders the paper figure from these stage results.

## What this experiment produces

- `experiments/figure-13-reordering-methods/stages/locality-aware-coaccess-gap/data/paper/figure_locality_aware_coaccess_gap.csv`

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

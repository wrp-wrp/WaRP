# Stage: offline reordering cost sweep

**Spec id:** `reorder-cost`
**Role:** stage of `figure-13-reordering-methods`
**Paper labels:** (owned by the parent Figure Experiment)
**Depends on:** (none)

## Directory layout

This directory is a measurement stage owned by
`experiments/figure-13-reordering-methods/`. It preserves frozen input data and a fresh
runner-output slot; the parent Figure Experiment owns the paper renderer.

| Path | What | State |
|---|---|---|
| `data/paper/` | authoritative frozen stage evidence | committed, SHA-256 hashed in `experiments/DATA_MANIFEST.json` |
| `data/latest/` | runner's fresh output data; written by `./run.sh` or `bash scripts/run_all.sh` (full mode) | gitignored |

To promote a fresh `data/latest/` run into the frozen stage slot, review the
result, copy `latest/` to `paper/`, then run `python3 -m artifact freeze`.

## What this experiment produces

Paper-facing outputs (`\includegraphics` targets):
- (none)

Frozen stage evidence:
- `experiments/figure-13-reordering-methods/stages/reorder-cost/data/paper/figure_reorder_cost_cluster_sweep.csv`

## Backend requirements

- Runs locally; no external services.

## Running

```bash
# Refresh this measurement stage:
./run.sh

# Render the paper figure from the parent owner:
../../render.sh light
```

Both scripts delegate to `python3 -m artifact`; commands come from
`artifact/registry.py`.

## Summary

Offline reordering runtime sweep across methods and cluster counts; feeds the figure-13-reordering-methods left panel.

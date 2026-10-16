# Intro: fine vs coarse clustering recall-matched comparison

**Spec id:** `table-01-cluster-coarsening`
**Role:** table-only Paper Experiment
**Paper labels:** tab:intro-coarse-cluster
**Depends on:** (none)

## Directory layout

This directory owns the introduction's experimental table and its frozen input data.

| Path | What | State |
|---|---|---|
| `*.csv` / `*.tex` (directly under this dir) | authoritative generated paper table | committed |
| `data/paper/` | authoritative frozen table input | committed, SHA-256 hashed in `experiments/DATA_MANIFEST.json` |
| `data/latest/` | runner's fresh output data; written by `./run.sh` or `bash scripts/run_all.sh` (full mode) | gitignored |
| `_regen/` / `_latest/` | tables rendered from frozen / fresh evidence | gitignored |

The render phase generates the derived CSV and exact TeX table used by the
paper. It intentionally has no figure.

## What this experiment produces

Paper-facing table artifacts:
- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex`

Expected files after render:
- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.csv`
- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex`

## Backend requirements

- Runs locally; no external services.

## Running

```bash
# Full reproduction: fresh measurement -> data/latest -> _latest
./run.sh
./render.sh

# Light profile (render-only from frozen data, ~seconds, no network):
./render.sh light
```

Both scripts delegate to `python3 -m artifact`; commands come from
`artifact/registry.py`.

## Summary

Recall-matched fine-vs-coarse clustering numbers backing tab:intro-coarse-cluster.

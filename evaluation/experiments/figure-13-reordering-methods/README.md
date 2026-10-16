# Reorder panels

**Paper figure:** 13

**Paper label:** `fig:reorder-panels`

Reproduces Figure 13: single-threaded permutation computation on the left
and locality quality on the right.

## Reproduce

```bash
./run.sh                 # run both stages and collect their outputs
./render.sh              # fresh data/latest -> _latest
./render.sh light        # frozen data/paper -> _regen
```

Light rendering reads `data/paper/stages/`; full rendering reads
`data/latest/stages/`. The two modes write to `_regen/` and `_latest/`.
From `evaluation/`, compare the frozen-data render with the paper copy:

```bash
python3 -m artifact compare --only figure-13-reordering-methods
```

## Measurement boundary

The left panel, “Permutation Computation”, reports the single-threaded time
to compute a cluster permutation. It excludes layout materialization and
payload writes. PCA methods keep the paper's PCA
`n`-D name; this controlled comparison fixes `n=2`.

## Frozen stages

- `data/paper/stages/reorder_cost.csv` comes from the `reorder-cost` Stage.
- `data/paper/stages/locality_gap.csv` comes from the
  `locality-aware-coaccess-gap` Stage.
- `experiments/DATA_MANIFEST.json` records both input hashes.

The full stage outputs are in `stages/reorder-cost/` and
`stages/locality-aware-coaccess-gap/`. The Figure 13 runner validates the new
outputs and copies the two plotting inputs to `data/latest/stages/`.

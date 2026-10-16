# Reorder panels

**Paper figure:** 13

**Paper label:** `fig:reorder-panels`

This Figure Experiment owns the committed composite reorder figure:
single-threaded Cluster Permutation computation on the left and locality
quality on the right.

## Reproduce

```bash
./run.sh                 # both stages plus owner finalizer
./render.sh              # fresh data/latest -> _latest
./render.sh light        # frozen data/paper -> _regen
```

Light rendering reads `data/paper/stages/`; full rendering reads the owner
finalizer's `data/latest/stages/`. Both write to safe regenerated slots. The
checked-in PDF is the committed paper artifact; the locally regenerated PDF is
pixel-identical.

## Measurement boundary

The left panel is deliberately titled “Permutation Computation”. It reports
the single-threaded time to compute the Cluster Permutation and excludes
Layout Materialization and payload writes. PCA methods keep the paper's PCA
`n`-D name; this controlled comparison fixes `n=2`.

## Frozen stages

- `data/paper/stages/reorder_cost.csv` comes from the `reorder-cost` Stage.
- `data/paper/stages/locality_gap.csv` comes from the
  `locality-aware-coaccess-gap` Stage.
- `experiments/DATA_MANIFEST.json` anchors both inputs.

The full Stage directories remain intact under `stages/reorder-cost/` and
`stages/locality-aware-coaccess-gap/` as measurement and analysis provenance.
The owner finalizer validates the live Stage outputs and publishes the two
smaller CSVs under `data/latest/stages/` for fresh rendering.

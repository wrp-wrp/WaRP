# Locality-aware co-access gap runner

This measurement stage computes the normalized co-access gap for each
reordering method and writes CSV/JSON outputs to
`experiments/figure-13-reordering-methods/stages/locality-aware-coaccess-gap/data/latest/`.
The `figure-13-reordering-methods` Paper Experiment owns the renderer and combines this
stage with the reorder-cost stage.

```bash
python3 -m artifact run locality-aware-coaccess-gap --dry-run
python3 -m artifact run locality-aware-coaccess-gap
python3 -m artifact render figure-13-reordering-methods
```

The runner never writes directly to a paper asset directory.

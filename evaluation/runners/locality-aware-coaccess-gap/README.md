# Locality-aware co-access gap runner

Computes the normalized co-access gap for each
reordering method and writes CSV/JSON outputs to
`experiments/figure-13-reordering-methods/stages/locality-aware-coaccess-gap/data/latest/`.
Figure 13 combines these results with the reorder-cost measurements.

```bash
python3 -m artifact run locality-aware-coaccess-gap --dry-run
python3 -m artifact run locality-aware-coaccess-gap
python3 -m artifact render figure-13-reordering-methods
```

Fresh results are written to `data/latest/`.

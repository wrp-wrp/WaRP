# Reordering cost

Measures the CPU time for computing cluster permutations across methods and
cluster counts. The Figure 13 experiment combines this stage with the
locality measurement.

Run from `evaluation/`:

```bash
python3 -m artifact run reorder-cost --dry-run
python3 -m artifact run reorder-cost
```

Fresh stage results go to
`experiments/figure-13-reordering-methods/stages/reorder-cost/data/latest/`.
The timed work excludes payload writes and layout materialization.

To collect both stages and render the complete Figure 13:

```bash
python3 -m artifact run figure-13-reordering-methods
python3 -m artifact render figure-13-reordering-methods
```

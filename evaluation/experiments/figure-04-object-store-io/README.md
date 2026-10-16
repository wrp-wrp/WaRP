# Object-store I/O microbenchmarks

**Paper figure:** 4

**Paper label:** `fig:bg-microbenchmarks`

Produces the two panels combined in Figure 4:

- `request_fanout.pdf`: request-count and fanout measurements.
- `surrogate_runtime_semantics.pdf`: calibration across four network topologies.

The supporting measurements are in `stages/real-s3-tiny-request-fanout-sanity/`
and `stages/s3-piecewise-calibration-curves/`. The inputs needed to render the
paper panels are in `data/paper/`.

## Reproduce

Run from this directory:

```bash
./run.sh                 # supporting measurements, then the four-topology run
./render.sh              # fresh data/latest -> _latest
./render.sh light        # frozen data/paper -> _regen
```

Rendering requires NumPy, Matplotlib, and SciPy. Live measurements require
four configured network topologies; see
[the runner guide](../../runners/figure-04-object-store-io/README.md).

Frozen input hashes are recorded in `experiments/DATA_MANIFEST.json`.

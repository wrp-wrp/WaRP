# Object-store I/O microbenchmarks

**Paper figure:** 4

**Paper label:** `fig:bg-microbenchmarks`

This Figure Experiment owns the two assets that LaTeX composes into one
top-level numbered figure:

- `request_fanout.pdf` for the request-count/fanout panel;
- `surrogate_runtime_semantics.pdf` for the four-topology calibration panel.

Subfigures are not separate Figure Experiments. Their reusable measurement
campaigns live below this owner as `stages/real-s3-tiny-request-fanout-sanity/`
and `stages/s3-piecewise-calibration-curves/`. The owner's `data/paper/`
contains the minimal frozen inputs used by the committed rendering; `stages/`
preserves the full measurement and analysis provenance.

## Reproduce

```bash
./run.sh                 # nested stages plus four-topology owner runner
./render.sh              # fresh data/latest -> _latest
./render.sh light        # frozen data/paper -> _regen
```

The render requires NumPy, Matplotlib, and SciPy. The owner run executes both
registered dependencies before collecting the four topology measurements;
topology configuration is documented under
`runners/figure-04-object-store-io/`.

## Provenance

The committed PDFs are the paper artifacts. Frozen input hashes are recorded
in the repository-wide `experiments/DATA_MANIFEST.json`, and light rendering
checks them without relying on a separate paper repository.

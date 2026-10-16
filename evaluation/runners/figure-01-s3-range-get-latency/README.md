# Figure 1: S3 Range-GET latency

Prepares the frozen latency breakdown used in the introduction figure.
Run from `evaluation/`:

```bash
python3 -m artifact run figure-01-s3-range-get-latency
python3 -m artifact render figure-01-s3-range-get-latency
```

The prepared CSV goes to
`experiments/figure-01-s3-range-get-latency/data/latest/data.csv`.
To render the paper's frozen measurements, use:

```bash
python3 -m artifact rebuild --only figure-01-s3-range-get-latency --data-slot paper
```

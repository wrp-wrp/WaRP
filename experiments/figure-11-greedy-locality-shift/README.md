# Greedy policy transfer

**Paper figure:** 11

**Paper label:** `fig:eval-greedy-locality-shift`

This Figure Experiment owns the three dumbbell panels comparing one
SIFT-tuned Greedy policy with WaRP across validation, locality, probe-count,
joint, and cross-dataset shifts.

## Reproduce

```bash
./run.sh                 # SIFT + MSong measurements -> data/latest
./render.sh              # fresh data/latest -> _latest
./render.sh light        # frozen data/paper -> _regen
```

The renderer validates both evidence files before writing
`figure_greedy_policy_transfer.pdf`, an ignored PNG preview, and a provenance
JSON file. The regenerated PDF is pixel-identical to the committed Figure 11
artifact; PDF container bytes differ because the historical export retained
creation metadata.

## Frozen evidence

- `data/paper/sift_summary.json` covers the four query-load shift cells.
- `data/paper/msong_summary.json` covers held-out MSong cross-dataset transfer.
- `experiments/DATA_MANIFEST.json` anchors both inputs.

## Environment

Rendering is local and requires Python, NumPy, and Matplotlib. `run.sh` executes
the heavyweight SIFT/MSong runner and requires the companion Lance flags and S3
inputs documented under `runners/figure-11-greedy-locality-shift/`.

## Intro Cluster Coarsening Sanity Check

This directory implements a very small introduction-side sanity check for the question:

> Why not simply make IVF clusters larger?

This is intentionally not a full evaluation. Its role is only to show that coarse clustering is not a free substitute for request planning.

The experiment is minimal by design:

- `1` dataset
- `2` IVF granularities
- `1` recall-matched `nprobe` choice per granularity
- a few hundred queries are enough
- final output: one small table

### Intended claim

The only claim this experiment should support is:

> Coarsening clusters can reduce requests/query, but it increases candidate processing, so it is not a clean way to remove the problem.

### Directory structure

- `run_data.py`
  - `tune`: local FAISS-based helper that searches for a recall-matched fine/coarse pair and records recall plus candidates/query
  - `plan`: freezes the final two-row manifest and optionally emits two runnable pipeline configs
  - `summarize`: combines the frozen ANN-side metrics from `tune` with run-time metrics from completed pipeline runs
- `generate.py`
  - renders the final intro table as CSV and LaTeX
- `experiments/table-01-cluster-coarsening/config/base.yaml`
  - experiment-owned WARP-IVF template used only by `plan --emit-configs`
- `experiments/table-01-cluster-coarsening/data/`
  - frozen tuning output, manifest, and final summary

### Recommended minimal setup

Dataset:

- `sift1m`

Configs:

- fine-grained: `nlist=4096`
- coarse-grained: `nlist=1024`

Primary table columns:

- `nlist`
- `nprobe`
- `Recall@10`
- `Requests/query`
- `Candidates/query`

### Workflow

1. Find a recall-matched pair.

```bash
python3 runners/table-01-cluster-coarsening/run_data.py \
  tune \
  --dataset sift1m \
  --fine-nlist 4096 \
  --coarse-nlist 1024 \
  --fine-nprobes 128,160,192,224,256 \
  --coarse-nprobes 32,48,64,80,96,112,128 \
  --queries 300 \
  --query-seed 12345 \
  --train-size 100000 \
  --recall-k 10
```

2. Freeze the two-row design and optionally emit configs.

```bash
python3 runners/table-01-cluster-coarsening/run_data.py \
  plan \
  --dataset sift1m \
  --fine-nlist 4096 \
  --coarse-nlist 1024 \
  --emit-configs
```

3. Run the two configs and fill the `run_dir` fields in:

- `experiments/table-01-cluster-coarsening/data/latest/manifest.json`

4. Summarize the final table inputs.

```bash
python3 runners/table-01-cluster-coarsening/run_data.py \
  summarize \
  --manifest experiments/table-01-cluster-coarsening/data/latest/manifest.json
```

Omitting `--manifest` is an offline reproduction path: it summarizes the
frozen paper manifest into `data/latest/` without changing the paper snapshot.

5. Render the intro table.

```bash
python3 runners/table-01-cluster-coarsening/generate.py
```

### Output files

After `summarize`:

- `experiments/table-01-cluster-coarsening/data/latest/summary.json`
- `experiments/table-01-cluster-coarsening/data/latest/intro_cluster_coarsening_sanity_check.csv`

After `generate.py`:

- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.csv`
- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex`

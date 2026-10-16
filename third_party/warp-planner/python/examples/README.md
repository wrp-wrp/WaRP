# Python examples

| File | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | Minimal `plan` → `evaluate_plan` flow. No dependency beyond the built binding. |
| [`faiss_end_to_end.py`](faiss_end_to_end.py) | The canonical use case: build a FAISS IVF index → pack clusters into a single blob → run a query → feed the cluster reads to `vecio_planner.plan(...)` → compare merged plan against the naive "one GET per cluster" baseline. |

## Setup

```bash
cd python
maturin develop --release      # builds and installs the vecio_planner extension
python examples/quickstart.py

# Optional FAISS end-to-end example:
pip install faiss-cpu numpy
python examples/faiss_end_to_end.py
```

Expected output (100k vectors, nlist=2048, nprobe=100):

```
Setup: 100000 vectors × 64 dims, IVF nlist=2048, nprobe=100
…
Query touches 100 clusters (= 100 S3 GETs naively).

  strategy        reqs   waste %   latency     $ per million queries
  ----------------------------------------------------------------------
  separate         100 req    0.0% waste   208.69 ms  $  40.00 / M-queries
  main_solution     24 req   87.4% waste    89.95 ms  $   9.60 / M-queries

MainSolution vs naive baseline:
  requests: 100 → 24 (76.0% fewer)
  latency: 208.69 → 89.95 ms (56.9% lower)
  S3 cost:  $30.40 saved per million queries
```

The high waste % comes from running on FAISS's default cluster_id
order — adjacent IDs aren't co-accessed, so any merge spans dead
bytes between probed clusters. A spatial reorder (PC1 sort, Hilbert)
of the cluster layout cuts waste dramatically while keeping the
request-count win. See `scripts/prepare-fixtures.sh` for the full
reorder pipeline and the `ivf_workload` Rust benchmark for
paper-scale numbers (sift1m, nlist=4096, nprobe=128, with reorder).

# Python examples

| Example | Purpose |
| --- | --- |
| [quickstart.py](quickstart.py) | Plan three ranges and evaluate the result |
| [faiss_end_to_end.py](faiss_end_to_end.py) | Build a FAISS IVF index and compare read plans |

## Setup

From `planner/`, activate a virtual environment and run:

```bash
CARGO_PROFILE_RELEASE_STRIP=none python -m pip install ./python
python python/examples/quickstart.py

# Optional FAISS example:
python -m pip install faiss-cpu numpy
python python/examples/faiss_end_to_end.py
```

The FAISS example reports model estimates for latency and cost. Its input has
100,000 vectors, 64 dimensions, `nlist=2048`, and `nprobe=100`. A recorded run
reduced 100 requests to 24, with modeled latency falling from 208.69 to 89.95 ms.
Results can vary with the FAISS version and cluster layout.

The example uses FAISS cluster-ID order, so merging can fetch substantial
unused data. See `planner/scripts/prepare-fixtures.sh` for the reordered
fixtures and the Rust `ivf_workload` benchmark for the SIFT1M setup.

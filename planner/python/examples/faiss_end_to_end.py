"""End-to-end IVF + vecio-planner demo.

This is what `vecio-planner` is *for*: cut the I/O fan-out of an IVF
vector search over cloud object storage. The example wires the
canonical pieces end-to-end in one script so you can see the full
shape:

    1. Build a FAISS IVF index over a small synthetic dataset.
    2. Pack cluster bytes into a single logical blob (one cluster =
       one contiguous byte range).
    3. Run a query → FAISS returns `nprobe` cluster IDs.
    4. Convert those IDs into `(cluster_id, offset, length)` reads —
       the input shape vecio-planner expects.
    5. Feed them to `vecio_planner.plan(..., "main_solution", ...)`.
    6. Compare against the naive "one GET per cluster" baseline:
       request count, predicted latency, $ per million queries.

Run with the PyO3 binding built into the active Python environment:

    cd python && maturin develop --release
    pip install faiss-cpu numpy
    python examples/faiss_end_to_end.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("faiss-cpu not installed: `pip install faiss-cpu`")

try:
    import vecio_planner as vp
except ImportError:
    sys.exit(
        "vecio_planner not built: `cd python && maturin develop --release`"
    )


# ----------------------------------------------------------------------
# Cost model — paper's two-segment S3 calibration + AWS GET pricing.
# ----------------------------------------------------------------------
PAPER_COST = dict(
    latency_model="piecewise_linear",
    latency_pw_breaks_bytes=[3_145_728],
    latency_pw_overheads_ms=[28.849557293569053, 6.266035367002434],
    latency_pw_rates_ms_per_byte=[2.824569398517584e-06, 1.0003677676972243e-05],
    latency_per_request=28.849557293569053,
    latency_per_byte=2.824569398517584e-06,
    concurrent_parallelism=16.0,
    cost_per_request=0.0000004,  # AWS S3 us-east-1: $0.0004 per 1 000 GETs
    cost_per_byte=0.0,
)


def build_ivf_index(n: int, d: int, nlist: int, seed: int = 42):
    """Train a FAISS IVFFlat index on `n` random vectors of dim `d`."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d), dtype=np.float32)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(base)
    index.add(base)
    return index, base


def pack_clusters_into_blob(index, cluster_bytes: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign each cluster a contiguous byte range in a single blob.

    Returns `(offsets, lengths)`, each indexed by `cluster_id`. This is
    the layout `vecio-planner` is built for: many small clusters laid
    out one after another in a single S3 object.
    """
    nlist = index.nlist
    invlists = index.invlists
    sizes = np.array([invlists.list_size(c) for c in range(nlist)], dtype=np.int64)
    lengths = sizes * cluster_bytes
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    return offsets, lengths


def query_io_requests(
    index, query: np.ndarray, nprobe: int,
    offsets: np.ndarray, lengths: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Run one IVF query, return the `(cluster_id, offset, length)`
    tuples that the inverted-list scan would issue."""
    index.nprobe = nprobe
    _, coarse_ids = index.quantizer.search(query.reshape(1, -1), nprobe)
    picked = sorted(int(c) for c in coarse_ids[0] if c >= 0)
    return [(c, int(offsets[c]), int(lengths[c])) for c in picked]


def report(label: str, plan: dict, requests, kwargs: dict):
    metrics = vp.evaluate_plan(requests, plan, **kwargs)
    t_ub = metrics["predicted_latency_ms_t_ub"]
    cost_per_q = (
        kwargs["cost_per_request"] * metrics["request_count"]
        + kwargs["cost_per_byte"] * metrics["total_bytes"]
    )
    waste = (
        100 * metrics["wasted_bytes"] / metrics["total_bytes"]
        if metrics["total_bytes"] else 0.0
    )
    print(
        f"  {label:<14}"
        f"  {metrics['request_count']:>4} req"
        f"  {waste:>5.1f}% waste"
        f"  {t_ub:>7.2f} ms"
        f"  ${cost_per_q * 1_000_000:>7.2f} / M-queries"
    )


def main() -> int:
    # 1) Build a small IVF index. Real workloads use sift1m / glove /
    # openimage at nlist=4096; these numbers keep the demo to a few
    # seconds while still reproducing the merge wins.
    n, d, nlist, nprobe = 100_000, 64, 2048, 100
    cluster_bytes = 1024  # bytes per indexed vector after compression
    print(f"Setup: {n} vectors × {d} dims, IVF nlist={nlist}, nprobe={nprobe}")
    print(f"Each cluster takes `n_in_cluster × {cluster_bytes}` bytes in the blob.")

    index, _base = build_ivf_index(n, d, nlist)

    # 2) Pack clusters into one logical blob (offsets + lengths).
    offsets, lengths = pack_clusters_into_blob(index, cluster_bytes)
    blob_mb = (offsets[-1] + lengths[-1]) / (1024 * 1024)
    print(
        f"Cluster sizes: min={lengths.min()} max={lengths.max()}  "
        f"blob = {blob_mb:.1f} MiB"
    )

    # 3) Run one query — get the cluster IDs FAISS would scan.
    rng = np.random.default_rng(0)
    query = rng.standard_normal(d, dtype=np.float32)
    requests = query_io_requests(index, query, nprobe, offsets, lengths)
    print(f"\nQuery touches {len(requests)} clusters (= {len(requests)} S3 GETs naively).")

    # 4) Plan with vecio-planner. Compare separate vs main_solution
    # under the paper's S3 cost model.
    print("\n  strategy        reqs   waste %   latency     $ per million queries")
    print(f"  {'-' * 70}")
    sep_plan = vp.plan(requests, "separate", **PAPER_COST)
    report("separate", sep_plan, requests, PAPER_COST)
    main_plan = vp.plan(requests, "main_solution", **PAPER_COST)
    report("main_solution", main_plan, requests, PAPER_COST)

    # 5) Headline numbers vs the naive baseline.
    sep_m = vp.evaluate_plan(requests, sep_plan, **PAPER_COST)
    main_m = vp.evaluate_plan(requests, main_plan, **PAPER_COST)
    sep_cost = PAPER_COST["cost_per_request"] * sep_m["request_count"]
    main_cost = PAPER_COST["cost_per_request"] * main_m["request_count"]
    sep_latency = sep_m["predicted_latency_ms_t_ub"]
    main_latency = main_m["predicted_latency_ms_t_ub"]
    print(
        f"\nMainSolution vs naive baseline:\n"
        f"  requests: {sep_m['request_count']} → {main_m['request_count']} "
        f"({100 * (1 - main_m['request_count'] / sep_m['request_count']):.1f}% fewer)\n"
        f"  latency: {sep_latency:.2f} → {main_latency:.2f} ms "
        f"({100 * (1 - main_latency / sep_latency):.1f}% lower)\n"
        f"  S3 cost:  ${(sep_cost - main_cost) * 1_000_000:.2f} saved per million queries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

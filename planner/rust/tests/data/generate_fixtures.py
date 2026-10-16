#!/usr/bin/env python3
"""Generate realistic IoRequest fixtures from a FAISS IVF index.

Produces per-query sequences that match the access pattern of IVF
vector search over a flat cloud-object cluster store: each cluster is a
contiguous byte range in a merged blob, a query touches `nprobe`
clusters, and we emit those (offset, length) pairs sorted by offset.

The `--dataset` argument can be:

- `gaussian` (default): synthetic dataset in this script, no external
  deps beyond numpy + faiss-cpu.
- `sift1m`: looks for base.npy / queries.npy under --dataset-dir;
  useful when the paper's data-processed cache is available locally.

Output: one JSON file per query in `--out-dir`, matching the
`IoRequest` schema on the Rust side:

    [ { "cluster_id": int, "offset": int, "length": int }, ... ]

Also writes `fixtures_manifest.json` listing every file plus metadata
(nlist, nprobe, num_queries, dataset name, bytes-per-cluster).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:
    print("[error] faiss is required: pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)


SIFT1M_BASE_SHA256 = "4921953796bc066d3a4be1c6b26e32a27b8931080f21afe30824a63e65cb768d"
SIFT1M_QUERIES_SHA256 = "88e61604b7ad2efee410551fbf6fd800594f4008ac599e2f75fe5a58f569f7b9"


def _load_heavy_tail_synthetic(
    n: int, d: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A more adversarial distribution: vectors are drawn from a mixture
    of many narrow gaussians plus a heavy-tailed spread. After IVF,
    cluster sizes span multiple orders of magnitude, so gap distributions
    are continuous rather than bimodal — this is the regime where DP
    demonstrably beats the greedy heuristic.
    """
    rng = np.random.default_rng(seed)
    # Big, sparse mixture of centers.
    k_centers = 64
    centers = rng.standard_normal((k_centers, d), dtype=np.float32) * 20.0
    # Power-law cluster size selector: a few centers get many vectors, most get few.
    pick = rng.zipf(1.6, size=n).astype(np.int64) % k_centers
    base = centers[pick] + rng.standard_normal((n, d), dtype=np.float32) * 0.5
    queries = rng.standard_normal((256, d), dtype=np.float32) * 15.0
    return base.astype(np.float32), queries.astype(np.float32)


def _load_gaussian(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d), dtype=np.float32)
    # A few structured clusters to make FAISS's IVF assignments non-trivial.
    centers = rng.standard_normal((8, d), dtype=np.float32) * 4.0
    for i in range(n):
        base[i] += centers[i % 8]
    queries = rng.standard_normal((200, d), dtype=np.float32)
    return base, queries


def _load_sift(path: Path) -> tuple[np.ndarray, np.ndarray]:
    base = np.load(path / "base.npy", mmap_mode="r")
    queries = np.load(path / "queries.npy", mmap_mode="r")
    if base.shape != (1_000_000, 128) or base.dtype != np.float32:
        raise ValueError(f"unexpected SIFT1M base array: shape={base.shape}, dtype={base.dtype}")
    if queries.shape != (10_000, 128) or queries.dtype != np.float32:
        raise ValueError(
            f"unexpected SIFT1M query array: shape={queries.shape}, dtype={queries.dtype}"
        )
    return base, queries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="gaussian",
                    choices=("gaussian", "heavy_tail", "sift1m"))
    ap.add_argument("--dataset-dir", default="", help="Path to sift1m data dir (if --dataset sift1m).")
    # Defaults match the paper's sift1m closed-loop config:
    # n=1M × 128d, nlist=4096, nprobe=128 (see
    # runners/closed-loop-qps-latency-tradeoff/run_data.py in the
    # paper artifact).
    ap.add_argument("--n", type=int, default=1_000_000, help="Base size for synthetic dataset.")
    ap.add_argument("--d", type=int, default=128, help="Dimension for synthetic dataset.")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=128)
    ap.add_argument("--num-queries", type=int, default=64)
    ap.add_argument("--cluster-bytes", type=int, default=512,
                    help="Bytes per indexed vector in the merged cluster "
                         "blob. The default is one 128-dimensional "
                         "float32 SIFT vector.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--faiss-threads", type=int, default=16,
                    help="OpenMP threads used by FAISS training and search.")
    ap.add_argument("--reorder", default="none",
                    choices=("none", "pca", "hilbert"),
                    help="Cluster-layout reorder applied before assigning "
                         "byte offsets. `none` keeps FAISS's arbitrary "
                         "cluster_id order (baseline). `pca` sorts "
                         "clusters by their centroids' PC1 projection — "
                         "a 1-D spatial reorder that approximates the "
                         "paper's Hilbert reorder for this benchmark. "
                         "`hilbert` is not yet implemented.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_arrays = None
    if args.dataset == "gaussian":
        base, queries = _load_gaussian(args.n, args.d, args.seed)
    elif args.dataset == "heavy_tail":
        base, queries = _load_heavy_tail_synthetic(args.n, args.d, args.seed)
    else:
        if not args.dataset_dir:
            print("[error] --dataset-dir is required for sift1m", file=sys.stderr)
            return 1
        dataset_dir = Path(args.dataset_dir)
        base_path = dataset_dir / "base.npy"
        queries_path = dataset_dir / "queries.npy"
        base, queries = _load_sift(dataset_dir)
        source_arrays = {
            "base_file": base_path.name,
            "base_sha256": _sha256_file(base_path),
            "queries_file": queries_path.name,
            "queries_sha256": _sha256_file(queries_path),
        }
        expected_arrays = {
            "base_file": "base.npy",
            "base_sha256": SIFT1M_BASE_SHA256,
            "queries_file": "queries.npy",
            "queries_sha256": SIFT1M_QUERIES_SHA256,
        }
        if source_arrays != expected_arrays:
            raise ValueError(
                "SIFT1M arrays do not match the canonical converted inputs"
            )

    n, d = base.shape
    print(f"[fixtures] dataset={args.dataset} n={n} d={d} nlist={args.nlist} nprobe={args.nprobe}")

    # Train IVF.
    if args.faiss_threads <= 0:
        raise ValueError("--faiss-threads must be positive")
    faiss.omp_set_num_threads(int(args.faiss_threads))
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, int(args.nlist), faiss.METRIC_L2)
    index.cp.seed = int(args.seed)
    index.cp.niter = 10
    index.cp.nredo = 1
    index.cp.max_points_per_centroid = 256
    index.cp.min_points_per_centroid = 39
    train_size = min(n, 200_000)
    index.train(base[:train_size])
    index.add(base)
    index.nprobe = int(args.nprobe)

    # Pull cluster metadata. invlists.list_size(c) gives cluster size.
    invlists = index.invlists
    cluster_sizes = np.array(
        [invlists.list_size(c) for c in range(int(args.nlist))], dtype=np.int64
    )

    # Reorder step: decide a permutation of the cluster ids. The
    # permutation determines the byte layout of the merged blob —
    # clusters early in the permutation land at low byte offsets, their
    # neighbours in the permutation end up byte-adjacent. The paper's
    # Hilbert reorder puts spatially-close centroids at adjacent positions
    # in this permutation; we use PC1 of the centroid matrix as a cheap
    # 1-D proxy.
    centroids = quantizer.reconstruct_n(0, int(args.nlist))
    if args.reorder == "none":
        permutation = np.arange(int(args.nlist), dtype=np.int64)
    elif args.reorder == "pca":
        centered = centroids - centroids.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = (centered @ vt[0]).astype(np.float64)
        permutation = np.argsort(pc1).astype(np.int64)
    else:
        print(f"[error] reorder={args.reorder!r} not implemented here", file=sys.stderr)
        return 1
    # cluster_position[c] = position of cluster c in the permuted layout.
    cluster_position = np.zeros_like(permutation)
    cluster_position[permutation] = np.arange(int(args.nlist), dtype=np.int64)
    # Byte offsets follow the permuted order.
    sorted_sizes = cluster_sizes[permutation]
    sorted_offsets = np.concatenate([[0], np.cumsum(sorted_sizes)]) * int(args.cluster_bytes)
    # Map each cluster_id to its byte offset / length.
    cluster_offsets = np.zeros(int(args.nlist), dtype=np.int64)
    cluster_lengths = cluster_sizes * int(args.cluster_bytes)
    for pos, c in enumerate(permutation):
        cluster_offsets[c] = sorted_offsets[pos]

    # Simulate nprobe searches: for each query, FAISS's coarse index
    # returns the top-nprobe cluster ids. We emit IoRequests for those,
    # sorted by byte offset (which is the reorder-permuted order).
    nq = min(int(args.num_queries), queries.shape[0])
    sel_queries = queries[:nq]
    _, coarse_ids = quantizer.search(sel_queries, int(args.nprobe))

    manifest_entries = []
    for qi in range(nq):
        picked = [int(c) for c in coarse_ids[qi].tolist() if c >= 0]
        picked.sort(key=lambda c: int(cluster_offsets[c]))
        requests = []
        for c in picked:
            requests.append(
                {
                    "cluster_id": int(c),
                    "offset": int(cluster_offsets[c]),
                    "length": int(cluster_lengths[c]),
                }
            )
        fixture_path = out_dir / f"query_{qi:04d}.json"
        fixture_path.write_text(json.dumps(requests) + "\n", encoding="utf-8")
        manifest_entries.append(
            {
                "file": fixture_path.name,
                "query_id": qi,
                "cluster_count": len(requests),
                "first_offset": int(requests[0]["offset"]) if requests else 0,
                "last_end": int(requests[-1]["offset"] + requests[-1]["length"]) if requests else 0,
            }
        )

    manifest = {
        "schema_version": 2,
        "dataset": args.dataset,
        "n": int(n),
        "d": int(d),
        "index_type": "faiss.IndexIVFFlat",
        "metric": "L2",
        "train_size": int(train_size),
        "clustering": {
            "seed": int(index.cp.seed),
            "iterations": int(index.cp.niter),
            "restarts": int(index.cp.nredo),
            "max_points_per_centroid": int(index.cp.max_points_per_centroid),
            "min_points_per_centroid": int(index.cp.min_points_per_centroid),
            "faiss_threads": int(args.faiss_threads),
        },
        "nlist": int(args.nlist),
        "nprobe": int(args.nprobe),
        "reorder": args.reorder,
        "num_queries": int(nq),
        "cluster_bytes": int(args.cluster_bytes),
        "total_blob_bytes": int(sorted_offsets[-1]),
        "seed": int(args.seed),
        "toolchain": {
            "faiss": str(getattr(faiss, "__version__", "unknown")),
            "numpy": str(np.__version__),
        },
        "queries": manifest_entries,
    }
    if source_arrays is not None:
        manifest["source_arrays"] = source_arrays
    (out_dir / "fixtures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[fixtures] wrote {nq} per-query files + manifest to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

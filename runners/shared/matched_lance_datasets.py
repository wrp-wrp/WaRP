#!/usr/bin/env python3
"""Prepare the matched Random/Reordered IVF-PQ datasets used by Figures 7/8.

The production campaigns first trained one IVF model, then reused one PQ
codebook for both layouts.  This module preserves the campaign-specific
centroid staging as well as that matched-pair seam while parameterizing local
paths and the destination S3 prefix for public reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from artifact.campaigns import (
    FIGURE11_MATCHED_DATASETS,
    LAYOUT_SEED,
    MATCHED_DATASETS,
    MatchedDataset,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "processed"
DEFAULT_MANIFEST_ROOT = (
    ROOT / "experiments" / "figure-07-real-lance-metrics" / "data" / "latest" / "dataset-builds"
)
PCA_VARIANCE_THRESHOLD = 0.35
PCA_MAX_COMPONENTS = 8
MAX_ROWS_PER_GROUP = 16_384
MAX_ROWS_PER_FILE = 1_048_576
ANCHOR_COUNT = 257
PREPARABLE_DATASETS = {**MATCHED_DATASETS, **FIGURE11_MATCHED_DATASETS}


@dataclass(frozen=True)
class Preparation:
    spec: MatchedDataset
    base_path: Path
    queries_path: Path
    ground_truth_path: Path
    random_uri: str
    reordered_uri: str
    manifest_path: Path


def preparation_plan(
    datasets: Sequence[str],
    *,
    data_root: Path,
    s3_root: str,
    manifest_root: Path,
) -> list[Preparation]:
    """Resolve a preparation request without touching data, S3, or Lance."""
    root = s3_root.rstrip("/")
    unknown = sorted(set(datasets) - set(PREPARABLE_DATASETS))
    if unknown:
        raise ValueError(f"unknown matched dataset(s): {', '.join(unknown)}")
    if not root.startswith("s3://"):
        raise ValueError("--s3-root or VECIO_REAL_LANCE_S3_ROOT must be an s3:// URI")
    plan = []
    for key in datasets:
        spec = PREPARABLE_DATASETS[key]
        local = data_root / spec.local_key
        plan.append(
            Preparation(
                spec=spec,
                base_path=local / "base.npy",
                queries_path=local / "queries.npy",
                ground_truth_path=local / "gt_I.npy",
                random_uri=f"{root}/{spec.random_object}",
                reordered_uri=f"{root}/{spec.reordered_object}",
                manifest_path=manifest_root / key / "manifest.json",
            )
        )
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_arrow(values: np.ndarray):
    import pyarrow as pa

    matrix = np.ascontiguousarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(matrix.reshape(-1), type=pa.float32()), matrix.shape[1]
    )


def _centroids_numpy(values, dimension: int) -> np.ndarray:
    return np.asarray(
        values.values.to_numpy(zero_copy_only=False), dtype=np.float32
    ).reshape(len(values), dimension)


def _multiset_sha256(values: np.ndarray) -> str:
    rows = sorted(hashlib.sha256(row.tobytes()).digest() for row in values)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row)
    return digest.hexdigest()


def _morton_nd(point: np.ndarray, bits_per_axis: int) -> int:
    axes = [int(value) for value in point]
    code = 0
    for bit in range(bits_per_axis):
        for axis, value in enumerate(axes):
            code |= ((value >> bit) & 1) << (bit * len(axes) + axis)
    return code


def pca_nd_zorder(centroids: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Return the exact PCA n-D Z-order policy used by the production builders."""
    from sklearn.decomposition import PCA

    k_max = min(PCA_MAX_COMPONENTS, len(centroids), centroids.shape[1])
    pca = PCA(n_components=k_max, svd_solver="full")
    coordinates = pca.fit_transform(centroids)
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative = np.cumsum(ratios)
    selected = min(
        max(
            int(np.searchsorted(cumulative, PCA_VARIANCE_THRESHOLD, side="left")) + 1,
            min(2, k_max),
        ),
        k_max,
    )
    bits_per_axis = min(32, max(1, 64 // selected))
    used = np.asarray(coordinates[:, :selected], dtype=np.float64)
    minimum = used.min(axis=0)
    span = used.max(axis=0) - minimum
    span[span <= 1e-12] = 1.0
    maximum_integer = (1 << bits_per_axis) - 1
    quantized = np.clip(
        np.rint((used - minimum) / span * maximum_integer),
        0,
        maximum_integer,
    ).astype(np.uint64)
    codes = np.fromiter(
        (_morton_nd(row, bits_per_axis) for row in quantized),
        dtype=np.uint64,
        count=len(quantized),
    )
    permutation = np.argsort(codes, kind="mergesort").astype(np.int64, copy=False)
    return permutation, {
        "method": "pca_nd_zorder",
        "pca_solver": "full",
        "variance_threshold": PCA_VARIANCE_THRESHOLD,
        "max_components": PCA_MAX_COMPONENTS,
        "selected_components": selected,
        "selected_cumulative_variance": float(cumulative[selected - 1]),
        "bits_per_axis": bits_per_axis,
        "permutation_sha256": hashlib.sha256(
            permutation.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def production_centroid_layouts(
    trained: np.ndarray,
    spec: MatchedDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    """Reconstruct the centroid staging used by each production campaign.

    The one-million-row Figure 7 datasets were built in two stages: first
    publishing a seeded-random source index, then training PQ and running
    PCA/Z-order on the centroids read back from that random source.  Applying
    PCA directly to the pre-randomization order is observably different, so retain the
    composition here.  The 100M campaign instead treated the source order as
    the PQ-training order and applied random/PCA layouts from that source.
    Figure 11 has no committed source builder and is explicitly labelled a
    reconstructed replication.
    """
    random_order = np.random.default_rng(
        [spec.random_seed, spec.ordinal]
    ).permutation(spec.nlist)
    strategy = spec.centroid_preparation
    if strategy == "legacy_random_source":
        source = trained[random_order]
        random_centroids = source
        random_record = {
            "method": "seeded_random_source",
            "seed": [spec.random_seed, spec.ordinal],
            "permutation_sha256": hashlib.sha256(
                random_order.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        }
    elif strategy == "legacy_source_random_target":
        source = trained
        random_centroids = source[random_order]
        random_record = {
            "method": "seeded_random_target",
            "seed": [spec.random_seed, spec.ordinal],
            "permutation_sha256": hashlib.sha256(
                random_order.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        }
    elif strategy == "reconstructed_source":
        source = trained
        random_centroids = source
        random_record = {
            "method": "source_order",
            "evidence_status": "reconstructed-replication",
        }
    else:
        raise ValueError(f"unsupported centroid preparation strategy: {strategy}")
    reorder_order, reorder_record = pca_nd_zorder(source)
    reorder_record = {
        **reorder_record,
        "source_stage": strategy,
    }
    return source, random_centroids, source[reorder_order], random_record, reorder_record


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_or_none(uri: str):
    import lance

    try:
        return lance.dataset(uri)
    except Exception:
        return None


def _write_dataset(uri: str, base: np.ndarray, dimension: int) -> None:
    import lance
    import pyarrow as pa

    schema = pa.schema(
        [("id", pa.int64()), ("vector", pa.list_(pa.float32(), dimension))]
    )
    chunk_rows = 100_000

    def batches() -> Iterable:
        for start in range(0, len(base), chunk_rows):
            stop = min(start + chunk_rows, len(base))
            vectors = np.ascontiguousarray(base[start:stop], dtype=np.float32)
            vector_array = pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), type=pa.float32()), dimension
            )
            yield pa.record_batch(
                [pa.array(np.arange(start, stop, dtype=np.int64)), vector_array],
                schema=schema,
            )

    lance.write_dataset(
        batches(),
        uri,
        schema=schema,
        mode="create",
        max_rows_per_group=MAX_ROWS_PER_GROUP,
        max_rows_per_file=MAX_ROWS_PER_FILE,
    )


def _validate_local_array(path: Path, *, columns: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2:
        raise ValueError(f"expected a two-dimensional array: {path} shape={values.shape}")
    if columns is not None and values.shape[1] != columns:
        raise ValueError(
            f"dimension mismatch: {path} shape={values.shape}, expected columns={columns}"
        )
    return values


def _validate_ground_truth(
    path: Path,
    *,
    query_rows: int,
    base_rows: int,
    k: int,
) -> np.ndarray:
    values = _validate_local_array(path)
    if values.shape[0] != query_rows or values.shape[1] < k:
        raise ValueError(
            f"ground-truth shape mismatch: {path} shape={values.shape}; "
            f"expected ({query_rows}, >= {k})"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"ground-truth IDs must be integers: {path} dtype={values.dtype}")
    top_k = np.asarray(values[:, :k], dtype=np.int64)
    if np.any(top_k < 0) or np.any(top_k >= base_rows):
        raise ValueError(
            f"ground-truth IDs are outside [0, {base_rows}): {path}"
        )
    if k > 1 and np.any(np.diff(np.sort(top_k, axis=1), axis=1) == 0):
        raise ValueError(f"ground-truth top-{k} contains duplicate IDs: {path}")
    return values


def ensure_ground_truth(
    base_path: Path,
    queries_path: Path,
    output_path: Path,
    *,
    k: int = 10,
    threads: int = 0,
) -> Path:
    """Compute exact L2 neighbors when a public source does not ship them.

    FAISS owns the large in-memory index; arrays remain memory-mapped while
    vectors and queries are fed in bounded batches.  Existing ground truth is
    validated before reuse, and new results are atomically published only
    after their complete shape and ID domain have been checked.
    """
    if k <= 0:
        raise ValueError("ground-truth k must be positive")
    base = _validate_local_array(base_path)
    queries = _validate_local_array(queries_path, columns=base.shape[1])
    if len(base) < k:
        raise ValueError(f"ground-truth k={k} exceeds base rows={len(base)}")
    if output_path.is_file():
        _validate_ground_truth(
            output_path,
            query_rows=len(queries),
            base_rows=len(base),
            k=k,
        )
        return output_path
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is required to compute missing exact ground truth"
        ) from exc
    if threads > 0:
        faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(base.shape[1])
    add_rows = 100_000
    for start in range(0, len(base), add_rows):
        stop = min(start + add_rows, len(base))
        index.add(np.ascontiguousarray(base[start:stop], dtype=np.float32))
        print(f"[ground-truth] indexed {stop}/{len(base)} rows", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        neighbors = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.int64,
            shape=(len(queries), k),
        )
        query_batch_rows = 64
        for start in range(0, len(queries), query_batch_rows):
            stop = min(start + query_batch_rows, len(queries))
            _, ids = index.search(
                np.ascontiguousarray(queries[start:stop], dtype=np.float32), k
            )
            neighbors[start:stop] = ids
        neighbors.flush()
        del neighbors
        _validate_ground_truth(
            temporary,
            query_rows=len(queries),
            base_rows=len(base),
            k=k,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[ground-truth] wrote exact top-{k}: {output_path}", flush=True)
    return output_path


def _build_index(uri: str, spec: MatchedDataset, centroids: np.ndarray, pq_codebook) -> None:
    import lance

    lance.dataset(uri).create_index(
        column="vector",
        index_type="IVF_PQ",
        num_partitions=spec.nlist,
        num_sub_vectors=spec.num_sub_vectors,
        metric="l2",
        ivf_centroids=_to_arrow(centroids),
        pq_codebook=pq_codebook,
        replace=True,
    )


def _deterministic_positions(rows: int) -> list[int]:
    positions = {0, 1, rows // 2, rows - 2, rows - 1}
    rng = np.random.default_rng(LAYOUT_SEED)
    while len(positions) < min(ANCHOR_COUNT, rows):
        positions.add(int(rng.integers(0, rows)))
    return sorted(positions)


def _anchor_digest(dataset, base: np.ndarray, dimension: int) -> str:
    positions = _deterministic_positions(len(base))
    table = dataset.take(positions, columns=["id", "vector"])
    ids = np.asarray(
        table.column("id").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    vectors_column = table.column("vector").combine_chunks()
    vectors = np.asarray(
        vectors_column.values.to_numpy(zero_copy_only=False), dtype=np.float32
    ).reshape(len(ids), dimension)
    expected_ids = np.asarray(positions, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"anchor ID mismatch: expected {expected_ids}, observed {ids}")
    if not np.array_equal(vectors, np.asarray(base[ids], dtype=np.float32)):
        raise ValueError("anchor vector mismatch against the public base array")
    digest = hashlib.sha256()
    digest.update(ids.astype("<i8", copy=False).tobytes())
    digest.update(vectors.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _index_record(dataset, item: Preparation, layout: str) -> dict[str, object]:
    indices = dataset.list_indices()
    if len(indices) != 1 or indices[0].get("type") != "IVF_PQ":
        raise ValueError(f"unexpected {layout} index for {item.spec.key}: {indices}")
    index_name = indices[0].get("name")
    if not isinstance(index_name, str) or not index_name:
        raise ValueError(f"missing logical index name for {layout} {item.spec.key}")
    stats_api = getattr(dataset, "stats", None)
    if stats_api is not None and hasattr(stats_api, "index_stats"):
        stats = stats_api.index_stats(index_name)
    else:
        stats = dataset.index_statistics(index_name)
    physical = stats.get("indices") if isinstance(stats, dict) else None
    if not isinstance(physical, list) or len(physical) != 1:
        raise ValueError(
            f"expected one physical index for {layout} {item.spec.key}: {stats}"
        )
    detail = physical[0]
    sub_index = detail.get("sub_index", {}) if isinstance(detail, dict) else {}
    observed_m = sub_index.get("num_sub_vectors")
    observed_nlist = detail.get("num_partitions") if isinstance(detail, dict) else None
    expected_stats = {
        "index_type": "IVF_PQ",
        "num_indices": 1,
        "num_indexed_rows": item.spec.rows,
        "num_unindexed_rows": 0,
        "num_unindexed_fragments": 0,
    }
    mismatches = [
        key for key, value in expected_stats.items() if stats.get(key) != value
    ]
    if detail.get("index_type") != "IVF_PQ":
        mismatches.append("physical_index_type")
    if sub_index.get("index_type") != "PQ":
        mismatches.append("sub_index_type")
    if observed_nlist != item.spec.nlist:
        mismatches.append("num_partitions")
    if observed_m != item.spec.num_sub_vectors:
        mismatches.append("num_sub_vectors")
    if sub_index.get("nbits") != 8:
        mismatches.append("nbits")
    if mismatches:
        raise ValueError(
            f"observed index contract differs for {layout} {item.spec.key}: "
            + ", ".join(sorted(set(mismatches)))
        )
    index_uuid = detail.get("uuid")
    if not isinstance(index_uuid, str) or not index_uuid:
        raise ValueError(f"missing physical index UUID for {layout} {item.spec.key}")
    return {
        "uri": item.random_uri if layout == "random" else item.reordered_uri,
        "dataset_version": int(dataset.version),
        "logical_index_name": index_name,
        "physical_index_uuid": index_uuid,
        "index_type": detail["index_type"],
        "nlist": int(observed_nlist),
        "num_sub_vectors": int(observed_m),
        "bits_per_subcode": int(sub_index["nbits"]),
    }


def _validate_pair(item: Preparation, base: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    datasets = {
        "random": _dataset_or_none(item.random_uri),
        "reordered": _dataset_or_none(item.reordered_uri),
    }
    if any(dataset is None for dataset in datasets.values()):
        raise ValueError("matched pair is incomplete")
    records: dict[str, object] = {}
    centroid_sets = {}
    anchor_hashes = {}
    for layout, dataset in datasets.items():
        assert dataset is not None
        if dataset.count_rows() != item.spec.rows:
            raise ValueError(f"row-count mismatch for {layout} {item.spec.key}")
        record = _index_record(dataset, item, layout)
        centroids = _centroids_numpy(dataset.centroids(column="vector"), item.spec.dimension)
        if centroids.shape != (item.spec.nlist, item.spec.dimension):
            raise ValueError(f"centroid shape mismatch for {layout} {item.spec.key}")
        centroid_sets[layout] = _multiset_sha256(centroids)
        anchor_hashes[layout] = _anchor_digest(dataset, base, item.spec.dimension)
        records[layout] = {
            **record,
            "centroid_multiset_sha256": centroid_sets[layout],
            "anchor_id_and_vector_sha256": anchor_hashes[layout],
        }
    if len(set(centroid_sets.values())) != 1:
        raise ValueError(f"Random/Reordered centroid sets differ for {item.spec.key}")
    if len(set(anchor_hashes.values())) != 1:
        raise ValueError(f"Random/Reordered anchor rows differ for {item.spec.key}")
    return _centroids_numpy(datasets["random"].centroids(column="vector"), item.spec.dimension), records


def _prepare_one(
    item: Preparation,
    *,
    ground_truth_k: int,
    faiss_threads: int,
) -> dict[str, object]:
    base = _validate_local_array(item.base_path, columns=item.spec.dimension)
    if base.shape != (item.spec.rows, item.spec.dimension) or base.dtype != np.float32:
        raise ValueError(
            f"unexpected {item.spec.key} base identity: {base.shape} {base.dtype}; "
            f"expected {(item.spec.rows, item.spec.dimension)} float32"
        )
    queries = _validate_local_array(item.queries_path, columns=item.spec.dimension)
    queries_sha256 = _sha256(item.queries_path)
    if queries_sha256 != item.spec.expected_query_sha256:
        raise ValueError(
            f"unexpected {item.spec.key} query identity: {queries_sha256}; "
            f"expected {item.spec.expected_query_sha256}"
        )
    ensure_ground_truth(
        item.base_path,
        item.queries_path,
        item.ground_truth_path,
        k=ground_truth_k,
        threads=faiss_threads,
    )
    ground_truth = _validate_ground_truth(
        item.ground_truth_path,
        query_rows=len(queries),
        base_rows=len(base),
        k=ground_truth_k,
    )
    if len(queries) < 10_000:
        raise ValueError(f"{item.spec.key} requires at least 10,000 queries")

    random_ds = _dataset_or_none(item.random_uri)
    reordered_ds = _dataset_or_none(item.reordered_uri)
    if (random_ds is None) != (reordered_ds is None):
        raise ValueError(
            f"refusing partial matched pair for {item.spec.key}; both targets must exist or "
            "neither. Inspect and remove only the declared partial targets, or choose an "
            "empty VECIO_REAL_LANCE_S3_ROOT before retrying"
        )

    build_record: dict[str, object] = {"reused_existing_pair": random_ds is not None}
    if random_ds is None:
        import lance
        from lance.indices.builder import IndicesBuilder
        from lance.indices.ivf import IvfModel

        print(f"[data] {item.spec.key} -> {item.random_uri}", flush=True)
        _write_dataset(item.random_uri, base, item.spec.dimension)
        print(f"[data] {item.spec.key} -> {item.reordered_uri}", flush=True)
        _write_dataset(item.reordered_uri, base, item.spec.dimension)

        random_ds = lance.dataset(item.random_uri)
        print(f"[ivf] training {item.spec.key} nlist={item.spec.nlist}", flush=True)
        random_ds.create_index(
            column="vector",
            index_type="IVF_FLAT",
            num_partitions=item.spec.nlist,
            metric="l2",
            replace=True,
        )
        trained = _centroids_numpy(
            random_ds.centroids(column="vector"), item.spec.dimension
        )
        (
            pq_training_centroids,
            random_centroids,
            reordered_centroids,
            random_record,
            reorder_record,
        ) = production_centroid_layouts(trained, item.spec)
        print(
            f"[pq] training shared {item.spec.key} M={item.spec.num_sub_vectors}",
            flush=True,
        )
        pq_model = IndicesBuilder(random_ds, "vector").train_pq(
            IvfModel(_to_arrow(pq_training_centroids), "l2"),
            item.spec.num_sub_vectors,
        )
        _build_index(item.random_uri, item.spec, random_centroids, pq_model.codebook)
        _build_index(
            item.reordered_uri,
            item.spec,
            reordered_centroids,
            pq_model.codebook,
        )
        build_record.update(
            {
                "base_path": str(item.base_path),
                "base_sha256": _sha256(item.base_path),
                "centroid_preparation": item.spec.centroid_preparation,
                "random_layout": random_record,
                "reordered_layout": reorder_record,
                "shared_pq_codebook_sha256_at_build": hashlib.sha256(
                    np.asarray(
                        pq_model.codebook.values.to_numpy(zero_copy_only=False),
                        dtype="<f4",
                    ).tobytes()
                ).hexdigest(),
            }
        )

    _, index_records = _validate_pair(item, base)
    if build_record["reused_existing_pair"]:
        if not item.manifest_path.is_file():
            raise ValueError(
                f"existing matched pair for {item.spec.key} has no local build manifest: "
                f"{item.manifest_path}; for externally managed objects, provide equivalent "
                "complete identity manifests directly to the measurement runner with "
                "--index-manifest-root, or choose an empty writable S3 prefix"
            )
        prior = json.loads(item.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(prior, dict):
            raise ValueError(f"invalid build manifest object: {item.manifest_path}")
        expected = {
            "status": "complete",
            "dataset": item.spec.key,
            "rows": item.spec.rows,
            "dimension": item.spec.dimension,
            "nlist": item.spec.nlist,
            "num_sub_vectors": item.spec.num_sub_vectors,
            "centroid_preparation": item.spec.centroid_preparation,
            "base_sha256": _sha256(item.base_path),
            "queries_sha256": queries_sha256,
            "ground_truth_sha256": _sha256(item.ground_truth_path),
        }
        mismatches = [
            key for key, value in expected.items() if prior.get(key) != value
        ]
        if prior.get("indices") != index_records:
            mismatches.append("indices")
        codebook_hash = prior.get("shared_pq_codebook_sha256_at_build")
        if (
            not isinstance(codebook_hash, str)
            or len(codebook_hash) != 64
            or any(character not in "0123456789abcdef" for character in codebook_hash)
        ):
            mismatches.append("shared_pq_codebook_sha256_at_build")
        if mismatches:
            raise ValueError(
                f"existing build manifest differs for {item.spec.key}: "
                + ", ".join(sorted(set(mismatches)))
            )
        print(f"[reuse] validated managed matched pair {item.spec.key}", flush=True)
        return prior
    payload = {
        "schema_version": 1,
        "status": "complete",
        "dataset": item.spec.key,
        "rows": item.spec.rows,
        "dimension": item.spec.dimension,
        "metric": "l2",
        "index_type": "IVF_PQ",
        "nlist": item.spec.nlist,
        "num_sub_vectors": item.spec.num_sub_vectors,
        "bits_per_subcode": 8,
        "compression": item.spec.compression,
        "index_reuse_identity": "physical_index_uuid_plus_dataset_version",
        "queries_path": str(item.queries_path),
        "queries_sha256": queries_sha256,
        "ground_truth_path": str(item.ground_truth_path),
        "ground_truth_sha256": _sha256(item.ground_truth_path),
        "indices": index_records,
        "write_settings": {
            "max_rows_per_group": MAX_ROWS_PER_GROUP,
            "max_rows_per_file": MAX_ROWS_PER_FILE,
        },
        **build_record,
    }
    _write_json_atomic(item.manifest_path, payload)
    return payload


def _ensure_public_arrays(item: Preparation, *, download: bool) -> None:
    present = (item.base_path.is_file(), item.queries_path.is_file())
    if all(present):
        return
    if any(present):
        raise ValueError(
            f"partial processed dataset for {item.spec.key}: expected both "
            f"{item.base_path} and {item.queries_path}"
        )
    if not download:
        raise FileNotFoundError(
            f"processed {item.spec.key} arrays are missing; rerun with "
            "--download-public-data or use `python -m artifact prepare-dataset`"
        )
    from artifact.prepare_dataset import PREPARERS

    print(f"[public-data] preparing {item.spec.key}", flush=True)
    PREPARERS[item.spec.local_key](
        item.base_path.parent,
        chunk_rows=65_536,
        download=True,
    )


def _default_s3_root() -> str:
    explicit = str(os.environ.get("VECIO_REAL_LANCE_S3_ROOT", "")).strip()
    if explicit:
        return explicit
    bucket = str(os.environ.get("VECIO_S3_BUCKET", "")).strip()
    return f"s3://{bucket}/real-lance-format-sanity" if bucket else ""


def _parse_dataset_list(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        # "all" means the seven-dataset Figure 7/8 campaign.  Figure 11 has
        # a distinct MSong M=60 contract and must be requested explicitly.
        return list(MATCHED_DATASETS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all", help="Comma-separated keys or 'all'.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--s3-root", default=_default_s3_root())
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--ground-truth-k", type=int, default=10)
    parser.add_argument("--faiss-threads", type=int, default=0)
    parser.add_argument(
        "--download-public-data",
        action="store_true",
        help="Download/convert missing public arrays before building the Lance pair.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print the deterministic plan without importing Lance, reading arrays, or touching S3.",
    )
    args = parser.parse_args(argv)
    try:
        plan = preparation_plan(
            _parse_dataset_list(args.datasets),
            data_root=args.data_root.expanduser().resolve(),
            s3_root=args.s3_root,
            manifest_root=args.manifest_root.expanduser().resolve(),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for item in plan:
        print(
            f"[dataset] {item.spec.key} local={item.base_path} "
            f"random={item.random_uri} reordered={item.reordered_uri}",
            flush=True,
        )
    if args.check_only:
        print(f"[ok] static matched-dataset plan contains {len(plan)} dataset(s)")
        return 0
    try:
        import lance  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matched dataset preparation requires the companion Lance Python package "
            "and pyarrow; run scripts/setup_lance_python.sh first"
        ) from exc
    for item in plan:
        _ensure_public_arrays(item, download=args.download_public_data)
        _prepare_one(
            item,
            ground_truth_k=args.ground_truth_k,
            faiss_threads=args.faiss_threads,
        )
    print(f"[ok] prepared or validated {len(plan)} matched dataset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

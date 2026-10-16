"""Reusable execution core for paper campaigns using ``full_pipeline_real_s3``.

Experiment runners own their schedule and paper-specific validation.  This
module owns the repetitive process boundary: environment construction,
resumable cell directories, raw-output validation, checksums, and repeat
aggregation.  Keeping that boundary here prevents Figures 7 and 8 from
carrying subtly different copies of the same benchmark harness.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


METRICS = ("qps", "p50_ms", "p95_ms", "p99_ms", "iops_per_q", "bytes_per_q")
AGGREGATE_METRICS = (*METRICS, "measured_payload_gbps")


@dataclass(frozen=True)
class Method:
    name: str
    layout: str
    strategy: str
    planner_p: int | None = None


@dataclass(frozen=True)
class Cell:
    experiment: str
    dataset: str
    method: Method
    repeat: int
    uri: str
    queries_npy: Path
    index_manifest: Path
    query_concurrency: int
    io_limit: int
    nprobe: int
    refine_factor: int
    k: int
    num_queries: int
    warmup_queries: int
    process_io_limit: int | None = None
    max_merge_span_bytes: int = 524_288
    index_cache_mb: int | None = None
    measured_passes: int = 1
    disable_partition_cache: bool = True
    network_interface: str | None = None
    order_ordinal: int | None = None
    runner_sha256: str | None = None
    campaign_contract_sha256: str | None = None

    @property
    def key(self) -> str:
        cache = "" if self.index_cache_mb is None else f"_cache{self.index_cache_mb}m"
        return (
            f"r{self.repeat}_{self.dataset}_{self.method.name}"
            f"_q{self.query_concurrency}_io{self.io_limit}{cache}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_identity(
    manifest_path: Path,
    *,
    dataset: str,
    layout: str,
    uri: str,
) -> dict[str, object]:
    """Bind a measurement cell to one validated physical Lance index."""
    path = manifest_path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"index build manifest is not an object: {path}")
    record_key = "reordered" if layout == "reorder" else layout
    indices = payload.get("indices")
    record = indices.get(record_key) if isinstance(indices, dict) else None
    mismatches = []
    if payload.get("status") != "complete":
        mismatches.append("status")
    if payload.get("dataset") != dataset:
        mismatches.append("dataset")
    if not isinstance(record, dict):
        mismatches.append(f"indices.{record_key}")
        record = {}
    from artifact.campaigns import MATCHED_DATASETS

    spec = MATCHED_DATASETS.get(dataset)
    if spec is None:
        mismatches.append("unknown_campaign_dataset")
        expected_nlist = None
        expected_m = None
    else:
        expected_nlist = spec.nlist
        expected_m = spec.num_sub_vectors
    if payload.get("nlist") != expected_nlist:
        mismatches.append("nlist")
    if payload.get("num_sub_vectors") != expected_m:
        mismatches.append("num_sub_vectors")
    if payload.get("bits_per_subcode") != 8:
        mismatches.append("bits_per_subcode")
    required_record = {
        "uri": uri,
        "index_type": "IVF_PQ",
        "nlist": expected_nlist,
        "num_sub_vectors": expected_m,
        "bits_per_subcode": 8,
    }
    mismatches.extend(
        f"indices.{record_key}.{key}"
        for key, value in required_record.items()
        if record.get(key) != value
    )
    for key in ("physical_index_uuid", "dataset_version"):
        if not record.get(key) and record.get(key) != 0:
            mismatches.append(f"indices.{record_key}.{key}")
    if mismatches:
        raise ValueError(
            f"index manifest does not own {dataset}/{record_key}: "
            + ", ".join(sorted(set(mismatches)))
        )
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256(path),
        "layout": record_key,
        "index": dict(record),
    }


def _observe_remote_index(uri: str, logical_index_name: str) -> dict[str, object]:
    import lance

    dataset = lance.dataset(uri)
    stats_api = getattr(dataset, "stats", None)
    if stats_api is not None and hasattr(stats_api, "index_stats"):
        stats = stats_api.index_stats(logical_index_name)
    else:
        stats = dataset.index_statistics(logical_index_name)
    physical = stats.get("indices") if isinstance(stats, dict) else None
    if not isinstance(physical, list) or len(physical) != 1:
        raise ValueError(f"expected one physical index at {uri}: {stats}")
    detail = physical[0]
    sub_index = detail.get("sub_index", {}) if isinstance(detail, dict) else {}
    return {
        "dataset_version": int(dataset.version),
        "logical_index_name": logical_index_name,
        "physical_index_uuid": detail.get("uuid"),
        "index_type": detail.get("index_type"),
        "nlist": detail.get("num_partitions"),
        "num_sub_vectors": sub_index.get("num_sub_vectors"),
        "bits_per_subcode": sub_index.get("nbits"),
    }


def observed_index_identity(
    manifest_path: Path,
    *,
    dataset: str,
    layout: str,
    uri: str,
) -> dict[str, object]:
    """Validate that the current remote object still matches its local manifest."""
    identity = index_identity(
        manifest_path,
        dataset=dataset,
        layout=layout,
        uri=uri,
    )
    record = identity["index"]
    assert isinstance(record, dict)
    logical_name = record.get("logical_index_name")
    if not isinstance(logical_name, str) or not logical_name:
        raise ValueError(f"manifest has no logical index name for {dataset}/{layout}")
    observed = _observe_remote_index(uri, logical_name)
    expected = {
        key: record.get(key)
        for key in (
            "dataset_version",
            "logical_index_name",
            "physical_index_uuid",
            "index_type",
            "nlist",
            "num_sub_vectors",
            "bits_per_subcode",
        )
    }
    if observed != expected:
        changed = [key for key in expected if observed.get(key) != expected.get(key)]
        raise ValueError(
            f"remote physical index differs for {dataset}/{layout}: "
            + ", ".join(changed)
        )
    return {**identity, "observed_remote_index": observed}


def _method_env(cell: Cell, base_env: Mapping[str, str] | None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    for name in (
        "LANCE_IO_MERGE_MAX_REQUESTS",
        "LANCE_IVF_BATCH_PREFETCH",
        "LANCE_IVF_STOCK_EXECUTION",
        "LANCE_IO_MERGE_CONCURRENT_PARALLELISM",
        "LANCE_DISABLE_PARTITION_CACHE",
        "LANCE_MAX_IOP_SIZE",
        "LANCE_PAYLOAD_PLAN_ONLY",
        "LANCE_IO_MERGE_EXACT_REQUESTS",
        "LANCE_IO_MERGE_LATENCY_PER_REQUEST_MS",
        "LANCE_IO_MERGE_LATENCY_PER_BYTE_MS",
        "LANCE_IO_MERGE_LATENCY_STD_MS",
    ):
        env.pop(name, None)
    env.update(
        {
            "LANCE_IO_THREADS": str(cell.io_limit),
            "LANCE_PROCESS_IO_THREADS_LIMIT": str(
                cell.io_limit if cell.process_io_limit is None else cell.process_io_limit
            ),
            "LANCE_IO_MERGE_STRATEGY": cell.method.strategy,
            "LANCE_IO_MERGE_MAX_SPAN_BYTES": str(cell.max_merge_span_bytes),
            "RAYON_NUM_THREADS": "1",
        }
    )
    if cell.method.planner_p is None:
        env["LANCE_IVF_STOCK_EXECUTION"] = "1"
    else:
        env["LANCE_IVF_BATCH_PREFETCH"] = "1"
        env["LANCE_IO_MERGE_CONCURRENT_PARALLELISM"] = str(cell.method.planner_p)
    if cell.disable_partition_cache:
        env["LANCE_DISABLE_PARTITION_CACHE"] = "1"
    return env


def _result_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return [dict(row) for row in payload]
    raise ValueError(f"unsupported benchmark result: {path}")


def validate_cell(cell: Cell, cell_dir: Path, binary: Path) -> dict:
    payload_files = ["meta.json", "result.json", "latencies.jsonl", "stdout", "stderr"]
    if cell.network_interface is not None:
        payload_files.append("bandwidth.json")
    required = (*payload_files, "sha256.json")
    missing = [name for name in required if not (cell_dir / name).is_file()]
    if missing:
        raise ValueError(f"{cell.key}: missing outputs: {missing}")
    recorded = json.loads((cell_dir / "sha256.json").read_text(encoding="utf-8"))
    for name in payload_files:
        if recorded.get(name) != sha256(cell_dir / name):
            raise ValueError(f"{cell.key}: checksum mismatch: {name}")
    meta = json.loads((cell_dir / "meta.json").read_text(encoding="utf-8"))
    recorded_cell = meta.get("cell", {})
    if not isinstance(recorded_cell, dict) or "binary" not in recorded_cell:
        raise ValueError(f"{cell.key}: invalid metadata envelope")
    if recorded_cell != _cell_metadata(cell, binary.resolve()):
        raise ValueError(f"{cell.key}: metadata does not match requested cell")
    rows = _result_rows(cell_dir / "result.json")
    if len(rows) != cell.measured_passes:
        raise ValueError(f"{cell.key}: expected {cell.measured_passes} measured pass(es)")
    result = rows[-1]
    process_io_limit = (
        cell.io_limit if cell.process_io_limit is None else cell.process_io_limit
    )
    expected = {
        "num_q": cell.num_queries,
        "Q": cell.query_concurrency,
        "R": process_io_limit,
    }
    if any(int(result[key]) != value for key, value in expected.items()):
        raise ValueError(f"{cell.key}: result identity mismatch")
    for metric in METRICS:
        value = float(result[metric])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{cell.key}: invalid {metric}={value}")
    elapsed_s = float(result["elapsed_s"])
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError(f"{cell.key}: invalid elapsed_s={elapsed_s}")
    result = dict(result)
    result["measured_payload_gbps"] = (
        float(result["bytes_per_q"]) * cell.num_queries * 8.0 / elapsed_s / 1e9
    )
    bandwidth = None
    if cell.network_interface is not None:
        bandwidth = json.loads((cell_dir / "bandwidth.json").read_text(encoding="utf-8"))
        if bandwidth.get("interface") != cell.network_interface:
            raise ValueError(f"{cell.key}: network-interface identity mismatch")
        for metric in ("rx_gbps", "tx_gbps"):
            value = float(bandwidth[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{cell.key}: invalid {metric}={value}")
            result[metric] = value
    latency_rows = [line for line in (cell_dir / "latencies.jsonl").read_text().splitlines() if line]
    if len(latency_rows) != cell.num_queries * cell.measured_passes:
        raise ValueError(f"{cell.key}: incomplete latency trace")
    row = {"cell": cell.key, "meta": meta["cell"], "result": result}
    if bandwidth is not None:
        row["bandwidth"] = bandwidth
    return row


def _cell_metadata(cell: Cell, binary: Path) -> dict:
    payload = asdict(cell)
    payload["method"] = asdict(cell.method)
    payload["queries_npy"] = str(cell.queries_npy.resolve())
    payload["index_manifest"] = str(cell.index_manifest.resolve())
    payload["index_identity"] = observed_index_identity(
        cell.index_manifest,
        dataset=cell.dataset,
        layout=cell.method.layout,
        uri=cell.uri,
    )
    payload["binary"] = str(binary.resolve())
    payload["binary_sha256"] = sha256(binary)
    payload["queries_sha256"] = sha256(cell.queries_npy)
    payload["shared_harness_sha256"] = sha256(Path(__file__).resolve())
    return payload


def run_cell(
    cell: Cell,
    *,
    binary: Path,
    cells_dir: Path,
    base_env: Mapping[str, str] | None = None,
    timeout_seconds: int = 14_400,
    cooldown_seconds: float = 0.0,
    force: bool = False,
) -> dict:
    binary = binary.resolve()
    cell_dir = cells_dir / cell.key
    if cell_dir.is_dir() and not force:
        try:
            row = validate_cell(cell, cell_dir, binary)
            print(f"[skip] valid cell {cell.key}", flush=True)
            return row
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            raise ValueError(
                f"{cell_dir} exists but is not a valid completed cell; "
                "inspect it or rerun with --force"
            )
    cell_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "result.json", "latencies.jsonl", "stdout", "stderr", "meta.json",
        "bandwidth.json", "sha256.json",
    ):
        path = cell_dir / name
        if path.exists():
            path.unlink()

    meta = {"schema_version": 1, "cell": _cell_metadata(cell, binary)}
    (cell_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    command = [
        str(binary),
        "--uri", cell.uri,
        "--queries-npy", str(cell.queries_npy.resolve()),
        "--num-queries", str(cell.num_queries),
        "--warmup-queries", str(cell.warmup_queries),
        "--reuse-warmup-queries",
        "--query-concurrency", str(cell.query_concurrency),
        "--nprobe", str(cell.nprobe),
        "--k", str(cell.k),
        "--refine-factor", str(cell.refine_factor),
        "--out-json", str(cell_dir / "result.json"),
        "--out-latencies-jsonl", str(cell_dir / "latencies.jsonl"),
    ]
    if cell.index_cache_mb is not None:
        command.extend(["--index-cache-mb", str(cell.index_cache_mb)])
    if cell.measured_passes != 1:
        command.extend(["--measured-passes", str(cell.measured_passes)])
    print(f"[run] {cell.key}", flush=True)
    counters_before = None
    whole_start_ns = None
    if cell.network_interface is not None:
        counters_before = _network_counters(cell.network_interface)
        whole_start_ns = time.monotonic_ns()
    with (cell_dir / "stdout").open("w") as stdout, (cell_dir / "stderr").open("w") as stderr:
        subprocess.run(
            command,
            check=True,
            env=_method_env(cell, base_env),
            stdout=stdout,
            stderr=stderr,
            timeout=timeout_seconds,
        )
    if cell.network_interface is not None:
        whole_end_ns = time.monotonic_ns()
        counters_after = _network_counters(cell.network_interface)
        assert counters_before is not None and whole_start_ns is not None
        elapsed_s = (whole_end_ns - whole_start_ns) / 1e9
        rx_bytes = counters_after[0] - counters_before[0]
        tx_bytes = counters_after[1] - counters_before[1]
        if elapsed_s <= 0 or rx_bytes < 0 or tx_bytes < 0:
            raise ValueError(f"{cell.key}: invalid network-counter boundary")
        bandwidth = {
            "schema_version": 1,
            "boundary": "whole benchmark process: dataset open, warm-up, and measured pass",
            "interface": cell.network_interface,
            "whole_process_elapsed_s": elapsed_s,
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
            "rx_gbps": rx_bytes * 8.0 / elapsed_s / 1e9,
            "tx_gbps": tx_bytes * 8.0 / elapsed_s / 1e9,
        }
        (cell_dir / "bandwidth.json").write_text(
            json.dumps(bandwidth, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    checksum_files = ["meta.json", "result.json", "latencies.jsonl", "stdout", "stderr"]
    if cell.network_interface is not None:
        checksum_files.append("bandwidth.json")
    checksums = {
        name: sha256(cell_dir / name)
        for name in checksum_files
    }
    (cell_dir / "sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    row = validate_cell(cell, cell_dir, binary)
    if cooldown_seconds > 0:
        time.sleep(cooldown_seconds)
    return row


def aggregate(
    rows: Iterable[dict],
    *,
    group_fields: tuple[str, ...],
    expected_repeats: int,
    metrics: tuple[str, ...] = AGGREGATE_METRICS,
) -> list[dict]:
    def field_value(row: dict, field: str) -> object:
        value = row["meta"][field]
        if field == "method" and isinstance(value, dict):
            return value["name"]
        return value

    groups: dict[tuple[object, ...], list[dict]] = {}
    for row in rows:
        key = tuple(field_value(row, field) for field in group_fields)
        groups.setdefault(key, []).append(row)
    output: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        if len(group) != expected_repeats:
            raise ValueError(f"group {key} has {len(group)} repeats, expected {expected_repeats}")
        aggregated_metrics = {}
        for metric in metrics:
            values = [float(row["result"][metric]) for row in group]
            aggregated_metrics[metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        output.append(
            {
                **dict(zip(group_fields, key)),
                "repeats": expected_repeats,
                "metrics": aggregated_metrics,
            }
        )
    return output


def _network_counters(interface: str) -> tuple[int, int]:
    """Read Linux interface counters used by the Figure 7 protocol."""
    if not interface or "/" in interface or interface in {".", ".."}:
        raise ValueError(f"invalid network interface: {interface!r}")
    root = Path("/sys/class/net") / interface / "statistics"
    try:
        return (
            int((root / "rx_bytes").read_text(encoding="ascii").strip()),
            int((root / "tx_bytes").read_text(encoding="ascii").strip()),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"cannot read network counters for {interface!r}") from exc

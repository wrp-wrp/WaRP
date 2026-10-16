#!/usr/bin/env python3
"""Rerun Figure 11's SIFT workload-shift and MSong transfer campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact.campaigns import (
    FIGURE11_PAPER_MSONG_INDEX_UUID,
    FIGURE11_PAPER_MSONG_OBJECT_ETAG,
    FIGURE11_PAPER_MSONG_OBJECT_SIZE,
    LANCE_BASE_COMMIT,
    PAPER_CAMPAIGNS,
)
from artifact.lance_toolchain import companion_root_for_binary, validate_lance_source


OUT = ROOT / "experiments" / HERE.name / "data" / "latest"
GAPS = "64,128,256,512,1024,2048,4096,8192,16384"
SPANS = "1,2,4,8,16,32"
SIFT = PAPER_CAMPAIGNS["figure11-sift-transfer"]
MSONG = PAPER_CAMPAIGNS["figure11-msong-transfer"]
MSONG_PROTOCOL_ID = "figure11-msong-transfer-v1"
SIFT_PROTOCOL_ID = "figure11-sift-transfer-v1"
PROTOCOL_SOURCES = (
    Path(__file__).resolve(),
    HERE / "sift_prepare.py",
    HERE / "sift_select_populations.py",
    HERE / "sift_validate.py",
    HERE / "sift_analyze.py",
    HERE / "msong_select_queries.py",
    HERE / "msong_build_manifests.py",
    HERE / "msong_validate.py",
    HERE / "msong_analyze.py",
    HERE / "paper_latency_model.json",
    ROOT / "src/cluster_reorder.py",
    ROOT / "artifact/campaigns.py",
)


def _lance_binary(example: str) -> Path:
    env_root = str(os.environ.get("VECIO_LANCE_ROOT", "")).strip()
    roots = ([Path(env_root).expanduser()] if env_root else []) + [
        ROOT.parent / "lance-s3-io-optimization",
        ROOT / "third_party" / "lance",
        ROOT.parent / "lance",
    ]
    paths = [root / "target/release/examples" / example for root in roots]
    return next((path for path in paths if path.is_file()), paths[0])


def _default_msong_uri() -> str:
    explicit = str(os.environ.get("VECIO_FIG11_MSONG_URI", "")).strip()
    if explicit:
        return explicit
    matched_root = str(os.environ.get("VECIO_REAL_LANCE_S3_ROOT", "")).strip()
    if matched_root:
        return f"{matched_root.rstrip('/')}/msong_pq_reorder.lance"
    bucket = str(os.environ.get("VECIO_S3_BUCKET", "")).strip()
    return f"s3://{bucket}/real-lance-format-sanity/msong_pq_reorder.lance" if bucket else ""


def _default_msong_index_identity() -> str:
    explicit = str(os.environ.get("VECIO_FIG11_MSONG_INDEX_IDENTITY", "")).strip()
    if explicit:
        return explicit
    return str(OUT / "dataset-builds" / "msong-figure11" / "manifest.json")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict-bin", default=str(_lance_binary("strict_budget_real_s3")))
    ap.add_argument(
        "--trace-export-bin",
        default=str(_lance_binary("full_pipeline_real_s3")),
        help="Figure-11 Lance binary containing --out-query-ranges-json.",
    )
    ap.add_argument("--sift-base", default=str(ROOT / "data/processed/sift1m/base.npy"))
    ap.add_argument("--sift-queries", default=str(ROOT / "data/processed/sift1m/queries.npy"))
    ap.add_argument("--msong-queries", default=str(ROOT / "data/processed/msong/queries.npy"))
    ap.add_argument("--msong-uri", default=_default_msong_uri())
    ap.add_argument(
        "--msong-index-identity",
        default=_default_msong_index_identity(),
        help=(
            "Managed build manifest or frozen paper inventory that pins the exact "
            "MSong index UUID used by the transfer stage."
        ),
    )
    ap.add_argument("--bucket", default=os.environ.get("VECIO_S3_BUCKET", ""))
    ap.add_argument("--region", default=os.environ.get("VECIO_S3_REGION", "ap-northeast-1"))
    ap.add_argument("--sift-prefix", default="warp-experiments/figure11-sift")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--timeout-seconds", type=int, default=3600)
    ap.add_argument("--cooldown-seconds", type=float, default=3.0)
    ap.add_argument(
        "--sift-input-root",
        default="",
        help="Reuse a prepared SIFT input root instead of rebuilding IVF-PQ inputs.",
    )
    ap.add_argument(
        "--msong-manifest-root",
        default="",
        help="Reuse prepared MSONG calibration/evaluation manifests instead of tracing Lance.",
    )
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _companion_git_metadata(binary: Path) -> dict[str, object]:
    try:
        root = companion_root_for_binary(binary)
        identity = validate_lance_source(root)
    except (ValueError, SystemExit) as exc:
        raise RuntimeError(f"cannot validate the paper Lance source: {exc}") from exc
    return {
        "root": str(root),
        "head": identity["lance_commit"],
        "dirty": False,
        "source_mode": identity["source_mode"],
        "source_tree_sha256": identity["source_tree_sha256"],
    }


def _toolchain_provenance(strict: Path, trace: Path) -> dict[str, object]:
    return {
        "paper_base_commit": LANCE_BASE_COMMIT,
        "binaries": {
            "strict_budget_real_s3": {
                "path": str(strict),
                "sha256": _sha256(strict),
                "source_checkout": _companion_git_metadata(strict),
            },
            "full_pipeline_real_s3": {
                "path": str(trace),
                "sha256": _sha256(trace),
                "source_checkout": _companion_git_metadata(trace),
            },
        },
        "protocol_sources": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in PROTOCOL_SOURCES
        },
    }


def _measurement_contract(
    args: argparse.Namespace,
    strict: Path,
    trace: Path,
) -> dict[str, object]:
    if args.sift_input_root:
        sift_inputs = {
            "mode": "prepared",
            "input_manifest_sha256": _sha256(
                Path(args.sift_input_root).expanduser().resolve() / "input_manifest.json"
            ),
        }
    else:
        sift_inputs = {
            "mode": "build",
            "base_sha256": _sha256(Path(args.sift_base).expanduser().resolve()),
            "queries_sha256": _sha256(Path(args.sift_queries).expanduser().resolve()),
            "bucket": args.bucket,
            "prefix": args.sift_prefix,
        }
    if args.msong_manifest_root:
        root = Path(args.msong_manifest_root).expanduser().resolve()
        msong_inputs = {
            "mode": "prepared",
            "calibration_sha256": _sha256(root / "calibration.json"),
            "evaluation_sha256": _sha256(root / "evaluation.json"),
            "input_identity_sha256": _sha256(root / "input_identity.json"),
        }
    else:
        identity = Path(args.msong_index_identity).expanduser().resolve()
        msong_inputs = {
            "mode": "trace",
            "queries_sha256": _sha256(Path(args.msong_queries).expanduser().resolve()),
            "dataset_uri": args.msong_uri,
            "index_identity_sha256": _sha256(identity),
        }
    return {
        "schema_version": 1,
        "toolchain": _toolchain_provenance(strict, trace),
        "region": args.region,
        "threads": args.threads,
        "sift_inputs": sift_inputs,
        "msong_inputs": msong_inputs,
    }


def _bind_work_directory(work: Path, contract: dict[str, object], *, force: bool) -> None:
    path = work / "measurement_contract.json"
    serialized = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != serialized and not force:
        raise SystemExit(
            f"Figure 11 work directory was created by a different input/toolchain: {work}; "
            "use a new --out-dir or rerun with --force"
        )
    if not path.is_file() and work.is_dir() and any(work.iterdir()) and not force:
        raise SystemExit(
            f"Figure 11 work directory has unbound prior outputs: {work}; "
            "use a new --out-dir or rerun with --force"
        )
    work.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _msong_index_contract(path: Path, expected_uri: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid MSong index identity object: {path}")
    expected_etag = None
    expected_size = None
    if "indices" in payload:
        indices = payload.get("indices")
        reordered = indices.get("reordered", {}) if isinstance(indices, dict) else {}
        uuid = reordered.get("physical_index_uuid")
        declared_uri = reordered.get("uri")
        expected = {
            "status": "complete",
            "dataset": "msong-figure11",
            "nlist": 4096,
            "num_sub_vectors": 60,
            "bits_per_subcode": 8,
        }
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        index_expected = {
            "uri": expected_uri,
            "index_type": "IVF_PQ",
            "nlist": 4096,
            "num_sub_vectors": 60,
            "bits_per_subcode": 8,
        }
        mismatches.extend(
            f"reordered.{key}"
            for key, value in index_expected.items()
            if reordered.get(key) != value
        )
        if mismatches:
            raise ValueError(
                "MSong managed manifest contract differs: "
                + ", ".join(sorted(mismatches))
            )
        mode = "reconstructed-replication"
    else:
        uuid = payload.get("expected_index_uuid")
        expected_etag = payload.get("payload_object_etag")
        expected_size = payload.get("payload_object_size")
        declared_uri = payload.get("dataset_uri")
        if declared_uri is None:
            bucket = payload.get("bucket")
            prefix = payload.get("prefix")
            if isinstance(bucket, str) and isinstance(prefix, str):
                declared_uri = f"s3://{bucket}/{prefix.strip('/')}"
        if declared_uri != expected_uri:
            raise ValueError(
                f"MSong inventory URI differs: {declared_uri!r} != {expected_uri!r}"
            )
        if not isinstance(expected_etag, str) or not expected_etag:
            raise ValueError("MSong inventory has no payload_object_etag")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError("MSong inventory has no positive payload_object_size")
        is_paper_object = (
            uuid == FIGURE11_PAPER_MSONG_INDEX_UUID
            and expected_etag == FIGURE11_PAPER_MSONG_OBJECT_ETAG
            and expected_size == FIGURE11_PAPER_MSONG_OBJECT_SIZE
            and declared_uri == expected_uri
        )
        mode = "paper-object" if is_paper_object else "external"
    if not isinstance(uuid, str) or not uuid:
        raise ValueError(f"MSong index identity has no pinned UUID: {path}")
    contract = {
        "mode": mode,
        "expected_index_uuid": uuid,
        "identity_path": str(path),
        "identity_sha256": _sha256(path),
    }
    if isinstance(expected_etag, str):
        contract["expected_object_etag"] = expected_etag
    if isinstance(expected_size, int):
        contract["expected_object_size"] = expected_size
    if mode == "reconstructed-replication":
        dataset_version = reordered.get("dataset_version")
        logical_name = reordered.get("logical_index_name")
        if not isinstance(dataset_version, int) or not isinstance(logical_name, str):
            raise ValueError(
                "MSong managed manifest lacks dataset_version/logical_index_name"
            )
        contract["expected_dataset_version"] = dataset_version
        contract["logical_index_name"] = logical_name
    return contract


def _bind_observed_payload_object(
    contract: dict[str, object],
    *,
    etag: str,
    size: int,
) -> dict[str, object]:
    expected_etag = contract.get("expected_object_etag")
    if isinstance(expected_etag, str) and etag != expected_etag:
        raise ValueError(
            f"MSONG payload ETag differs from the declared identity: {etag!r} != "
            f"{expected_etag!r}"
        )
    expected_size = contract.get("expected_object_size")
    if isinstance(expected_size, int) and size != expected_size:
        raise ValueError(
            f"MSONG payload size differs from the declared identity: {size} != "
            f"{expected_size}"
        )
    return {
        **contract,
        "observed_payload_object_etag": etag,
        "observed_payload_object_size": int(size),
    }


def _observe_msong_payload(
    dataset_uri: str,
    contract: dict[str, object],
    *,
    region: str,
) -> tuple[str, dict[str, object]]:
    parsed = urlparse(dataset_uri)
    uuid = str(contract["expected_index_uuid"])
    object_key = f"{parsed.path.lstrip('/').rstrip('/')}/_indices/{uuid}/auxiliary.idx"
    import boto3

    head = boto3.client("s3", region_name=region).head_object(
        Bucket=parsed.netloc,
        Key=object_key,
    )
    return object_key, _bind_observed_payload_object(
        contract,
        etag=str(head["ETag"]),
        size=int(head["ContentLength"]),
    )


def _validate_observed_managed_msong_index(
    dataset_uri: str,
    contract: dict[str, object],
) -> dict[str, object]:
    if contract.get("mode") != "reconstructed-replication":
        return {}
    import lance

    dataset = lance.dataset(dataset_uri)
    logical_name = str(contract["logical_index_name"])
    stats_api = getattr(dataset, "stats", None)
    if stats_api is not None and hasattr(stats_api, "index_stats"):
        stats = stats_api.index_stats(logical_name)
    else:
        stats = dataset.index_statistics(logical_name)
    physical = stats.get("indices") if isinstance(stats, dict) else None
    if not isinstance(physical, list) or len(physical) != 1:
        raise ValueError(f"managed MSong index statistics are invalid: {stats}")
    detail = physical[0]
    sub_index = detail.get("sub_index", {}) if isinstance(detail, dict) else {}
    observed = {
        "dataset_version": int(dataset.version),
        "physical_index_uuid": detail.get("uuid"),
        "index_type": detail.get("index_type"),
        "nlist": detail.get("num_partitions"),
        "num_sub_vectors": sub_index.get("num_sub_vectors"),
        "bits_per_subcode": sub_index.get("nbits"),
    }
    expected = {
        "dataset_version": contract["expected_dataset_version"],
        "physical_index_uuid": contract["expected_index_uuid"],
        "index_type": "IVF_PQ",
        "nlist": 4096,
        "num_sub_vectors": 60,
        "bits_per_subcode": 8,
    }
    if observed != expected:
        changed = [key for key in expected if observed.get(key) != expected.get(key)]
        raise ValueError(
            "managed MSong remote index differs: " + ", ".join(changed)
        )
    return observed


def _validate_msong_manifest_root(root: Path) -> dict[str, object]:
    paths = {
        "calibration": root / "calibration.json",
        "evaluation": root / "evaluation.json",
        "identity": root / "input_identity.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    identity = json.loads(paths["identity"].read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise ValueError("input_identity.json is not a JSON object")
    selected_uuid = identity.get("selected_index_uuid")
    dataset_uri = identity.get("dataset_uri")
    payload_uri = identity.get("payload_object_uri")
    if isinstance(dataset_uri, str) and isinstance(selected_uuid, str):
        expected_payload_uri = (
            f"{dataset_uri.rstrip('/')}/_indices/{selected_uuid}/auxiliary.idx"
        )
    else:
        expected_payload_uri = None
    expected = {
        "protocol_id": MSONG_PROTOCOL_ID,
        "calibration_manifest_sha256": _sha256(paths["calibration"]),
        "evaluation_manifest_sha256": _sha256(paths["evaluation"]),
        "warmup_queries": MSONG.warmup_queries,
        "calibration_queries": MSONG.measured_queries,
        "evaluation_queries": MSONG.measured_queries,
        "nlist": 4096,
        "nprobe": 64,
    }
    mismatches = [key for key, value in expected.items() if identity.get(key) != value]
    if not isinstance(dataset_uri, str) or not dataset_uri.startswith("s3://"):
        mismatches.append("dataset_uri")
    if not isinstance(selected_uuid, str) or not selected_uuid:
        mismatches.append("selected_index_uuid")
    if payload_uri != expected_payload_uri:
        mismatches.append("payload_object_uri")
    if not isinstance(identity.get("payload_object_etag"), str):
        mismatches.append("payload_object_etag")
    if not isinstance(identity.get("payload_object_size"), int) or int(
        identity.get("payload_object_size", 0)
    ) <= 0:
        mismatches.append("payload_object_size")
    object_size = int(identity.get("payload_object_size", 0))
    for name in ("calibration", "evaluation"):
        manifest = json.loads(paths[name].read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            mismatches.append(f"{name}_object")
            continue
        warmup_ranges = manifest.get("warmup_queries")
        measured_ranges = manifest.get("queries")
        if not isinstance(warmup_ranges, list) or len(warmup_ranges) != MSONG.warmup_queries:
            mismatches.append(f"{name}_warmup_queries")
        if not isinstance(measured_ranges, list) or len(measured_ranges) != MSONG.measured_queries:
            mismatches.append(f"{name}_queries")
        if manifest.get("nlist") != 4096 or manifest.get("nprobe") != 64:
            mismatches.append(f"{name}_geometry")
        if manifest.get("uri") != payload_uri:
            mismatches.append(f"{name}_uri")
        for section, query_ranges in (
            ("warmup_queries", warmup_ranges),
            ("queries", measured_ranges),
        ):
            if not isinstance(query_ranges, list):
                mismatches.append(f"{name}_{section}_ranges")
                continue
            invalid = False
            for ranges in query_ranges:
                if not isinstance(ranges, list) or not ranges:
                    invalid = True
                    break
                previous_start = -1
                for value in ranges:
                    if not isinstance(value, list) or len(value) != 2:
                        invalid = True
                        break
                    start, end = value
                    if (
                        not isinstance(start, int)
                        or not isinstance(end, int)
                        or start < 0
                        or end <= start
                        or end > object_size
                        or start < previous_start
                    ):
                        invalid = True
                        break
                    previous_start = start
                if invalid:
                    break
            if invalid:
                mismatches.append(f"{name}_{section}_ranges")
    if mismatches:
        raise ValueError("identity/manifest mismatch: " + ", ".join(sorted(set(mismatches))))
    return identity


def _validate_observed_msong_payload(
    identity: dict[str, object],
    *,
    region: str,
) -> dict[str, object]:
    """Require a precomputed MSONG identity to still name the same S3 bytes."""
    object_uri = str(identity["payload_object_uri"])
    parsed = urlparse(object_uri)
    import boto3

    head = boto3.client("s3", region_name=region).head_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )
    observed = {
        "payload_object_etag": str(head["ETag"]),
        "payload_object_size": int(head["ContentLength"]),
    }
    mismatches = [
        key for key, value in observed.items() if identity.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "MSONG precomputed payload identity differs from S3: "
            + ", ".join(mismatches)
        )
    return observed


def _validate_sift_input_root(root: Path, queries_path: Path) -> dict[str, object]:
    manifest_path = root / "input_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input_manifest.json is not a JSON object")
    mismatches = []
    if payload.get("protocol_id") != SIFT_PROTOCOL_ID:
        mismatches.append("protocol_id")
    if payload.get("queries_npy_sha256") != _sha256(queries_path):
        mismatches.append("queries_npy_sha256")
    fixed_files = {
        "faiss_index_sha256": root / "model/sift1m_ivf4096_pq64x8.faiss",
        "selection_manifest_sha256": root / "analysis/selection_manifest.json",
        "geometry_summary_sha256": root / "analysis/geometry_summary.json",
    }
    for key, path in fixed_files.items():
        if not path.is_file() or payload.get(key) != _sha256(path):
            mismatches.append(key)
    outputs = payload.get("outputs")
    expected_outputs = {
        f"inputs/nprobe_{nprobe:04d}/{population}/{layout}.json"
        for nprobe in (32, 256)
        for population in ("tight", "loose")
        for layout in ("original", "reordered")
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        mismatches.append("outputs")
    else:
        resolved_root = root.resolve()
        for relative, digest in outputs.items():
            path = (root / relative).resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError:
                mismatches.append(f"outputs.{relative}")
                continue
            if not path.is_file() or digest != _sha256(path):
                mismatches.append(f"outputs.{relative}")
    if mismatches:
        raise ValueError(
            "SIFT input manifest mismatch: " + ", ".join(sorted(set(mismatches)))
        )
    return payload


def _binary_has_flag(path: Path, flag: str) -> bool:
    """Check the compiled CLI contract without executing a cloud benchmark."""
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        result = subprocess.run(
            [str(path), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return flag in result.stdout


def _lance_contract_errors(strict: Path, trace: Path) -> list[str]:
    missing = []
    if not _binary_has_flag(strict, "--only-method"):
        missing.append(f"{strict}: --only-method")
    if not _binary_has_flag(trace, "--out-query-ranges-json"):
        missing.append(f"{trace}: --out-query-ranges-json")
    return missing


def _require_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    strict = Path(args.strict_bin).expanduser().resolve()
    trace = Path(args.trace_export_bin).expanduser().resolve()
    errors: list[str] = []
    for label, path in (("strict-budget", strict), ("trace-export", trace)):
        if not path.is_file() or not os.access(path, os.X_OK):
            errors.append(f"{label} binary is missing or not executable: {path}")
    if not errors:
        errors.extend(_lance_contract_errors(strict, trace))
    if not args.region:
        errors.append("--region or VECIO_S3_REGION is required")
    if args.sift_input_root:
        root = Path(args.sift_input_root).expanduser().resolve()
        try:
            _validate_sift_input_root(
                root,
                Path(args.sift_queries).expanduser().resolve(),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid --sift-input-root {root}: {exc}")
    else:
        if not args.bucket:
            errors.append("--bucket or VECIO_S3_BUCKET is required to prepare SIFT")
        for label, raw in (("SIFT base", args.sift_base), ("SIFT queries", args.sift_queries)):
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                errors.append(f"{label} is missing: {path}")
    if args.msong_manifest_root:
        root = Path(args.msong_manifest_root).expanduser().resolve()
        try:
            identity = _validate_msong_manifest_root(root)
            _validate_observed_msong_payload(identity, region=args.region)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid --msong-manifest-root {root}: {exc}")
    else:
        msong_queries = Path(args.msong_queries).expanduser().resolve()
        if not msong_queries.is_file():
            errors.append(f"MSong queries are missing: {msong_queries}")
        if not args.msong_uri.startswith("s3://"):
            errors.append("--msong-uri or VECIO_FIG11_MSONG_URI must be an s3:// URI")
        identity = Path(args.msong_index_identity).expanduser().resolve()
        if not identity.is_file():
            errors.append(
                "--msong-index-identity or VECIO_FIG11_MSONG_INDEX_IDENTITY "
                f"is missing: {identity}"
            )
        elif args.msong_uri.startswith("s3://"):
            try:
                contract = _msong_index_contract(identity, args.msong_uri)
                _validate_observed_managed_msong_index(args.msong_uri, contract)
                _observe_msong_payload(
                    args.msong_uri,
                    contract,
                    region=args.region,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid MSong index identity {identity}: {exc}")
    if errors:
        raise SystemExit("Figure 11 preflight failed:\n  - " + "\n  - ".join(errors))
    return strict, trace


def _run_strict(
    args: argparse.Namespace,
    *,
    label: str,
    dataset: str,
    original: Path,
    reordered: Path,
    latency_model: Path,
    out_dir: Path,
    nprobe: int,
    gaps: str,
    spans: str,
    only_method: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{label}.json"
    out_csv = out_dir / f"{label}.csv"
    if out_json.is_file() and out_csv.is_file() and not args.force:
        print(f"[skip] existing strict-budget cell {label}")
        return out_json
    campaign = MSONG if dataset == "msong" else SIFT
    command = [
        str(Path(args.strict_bin).expanduser().resolve()),
        "--dataset", dataset,
        "--original-query-json", str(original.resolve()),
        "--reordered-query-json", str(reordered.resolve()),
        "--latency-model-json", str(latency_model.resolve()),
        "--out-json", str(out_json),
        "--out-csv", str(out_csv),
        "--ks", "128",
        "--greedy-gap-kb", gaps,
        "--greedy-span-caps-mb", spans,
        "--num-queries", str(campaign.measured_queries),
        "--warmup-queries", str(campaign.warmup_queries),
        "--query-concurrency", str(campaign.query_concurrency),
        "--balanced-ps", "32",
        "--process-io-threads-limit", str(campaign.io_limit),
        "--cloud-io-parallelism", str(campaign.io_limit),
        "--max-iop-size-bytes", "549755813888",
    ]
    if only_method:
        command.extend(["--only-method", only_method])
    logs = out_dir.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"[run] {label}", flush=True)
    env = dict(os.environ)
    env["AWS_DEFAULT_REGION"] = args.region
    with (logs / f"{label}.stdout").open("w") as stdout, (logs / f"{label}.stderr").open("w") as stderr:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=args.timeout_seconds,
        )
    if args.cooldown_seconds > 0:
        time.sleep(args.cooldown_seconds)
    return out_json


def _prepare_sift(args: argparse.Namespace, work: Path) -> Path:
    if args.sift_input_root:
        root = Path(args.sift_input_root).expanduser().resolve()
        _validate_sift_input_root(
            root,
            Path(args.sift_queries).expanduser().resolve(),
        )
        return root
    root = work / "sift_inputs"
    manifest = root / "input_manifest.json"
    if not manifest.is_file() or args.force:
        _run(
            [
                sys.executable,
                str(HERE / "sift_prepare.py"),
                "--base-npy", str(Path(args.sift_base).expanduser().resolve()),
                "--queries-npy", str(Path(args.sift_queries).expanduser().resolve()),
                "--vecio-root", str(ROOT),
                "--output-root", str(root),
                "--bucket", args.bucket,
                "--region", args.region,
                "--prefix", args.sift_prefix,
                "--threads", str(args.threads),
                "--ivf-iters", "20",
                "--pq-iters", "20",
            ]
        )
        _run(
            [
                sys.executable,
                str(HERE / "sift_select_populations.py"),
                "--base-root", str(root),
                "--queries-npy", str(Path(args.sift_queries).expanduser().resolve()),
                "--output-root", str(root),
                "--threads", str(args.threads),
            ]
        )
    _validate_sift_input_root(
        root,
        Path(args.sift_queries).expanduser().resolve(),
    )
    return root


def _sift(args: argparse.Namespace, work: Path, latency_model: Path) -> tuple[Path, Path]:
    inputs = _prepare_sift(args, work)
    calibration_dir = work / "sift_calibration"
    results_dir = work / "sift_results"
    policy = work / "sift_policies_before_shift.json"
    summary = work / "sift_summary.json"
    calibration = _run_strict(
        args,
        label="calibration_tight_nprobe0032",
        dataset="sift1m",
        original=inputs / "inputs/nprobe_0032/tight/original.json",
        reordered=inputs / "inputs/nprobe_0032/tight/reordered.json",
        latency_model=latency_model,
        out_dir=calibration_dir,
        nprobe=32,
        gaps=GAPS,
        spans=SPANS,
    )
    _run(
        [
            sys.executable, str(HERE / "sift_validate.py"), str(calibration),
            "--expected-nlist", "4096", "--expected-nprobe", "32", "--expected-cells", "54",
        ]
    )
    if not policy.is_file() or args.force:
        _run(
            [
                sys.executable, str(HERE / "sift_analyze.py"), "freeze",
                "--calibration", str(calibration), "--output", str(policy),
            ]
        )
    frozen = json.loads(policy.read_text(encoding="utf-8"))
    gap = int(frozen["greedy"]["gap_kib"])
    span_mib = int(frozen["greedy"]["span_cap_bytes"]) // 2**20
    schedule = (
        ("tight", 32, 1), ("loose", 32, 1), ("tight", 256, 1), ("loose", 256, 1),
        ("loose", 256, 2), ("tight", 256, 2), ("loose", 32, 2), ("tight", 32, 2),
        ("loose", 32, 3), ("tight", 32, 3), ("loose", 256, 3), ("tight", 256, 3),
    )
    for population, nprobe, repeat in schedule:
        label = f"{population}_nprobe{nprobe:04d}_repeat_{repeat:02d}"
        result = _run_strict(
            args,
            label=label,
            dataset="sift1m",
            original=inputs / f"inputs/nprobe_{nprobe:04d}/{population}/original.json",
            reordered=inputs / f"inputs/nprobe_{nprobe:04d}/{population}/reordered.json",
            latency_model=latency_model,
            out_dir=results_dir,
            nprobe=nprobe,
            gaps=str(gap),
            spans=str(span_mib),
        )
        _run(
            [
                sys.executable, str(HERE / "sift_validate.py"), str(result),
                "--expected-nlist", "4096", "--expected-nprobe", str(nprobe),
                "--expected-cells", "1", "--expected-gap-kib", str(gap),
                "--expected-span-mib", str(span_mib),
            ]
        )
    _run(
        [
            sys.executable, str(HERE / "sift_analyze.py"), "final",
            "--results-root", str(results_dir), "--freeze", str(policy),
            "--input-manifest", str(inputs / "input_manifest.json"),
            "--geometry", str(inputs / "analysis/geometry_summary.json"),
            "--output", str(summary),
        ]
    )
    return summary, policy


def _prepare_msong(args: argparse.Namespace, work: Path) -> tuple[Path, Path]:
    if args.msong_manifest_root:
        manifests = Path(args.msong_manifest_root).expanduser().resolve()
        identity = manifests / "input_identity.json"
        try:
            identity_payload = _validate_msong_manifest_root(manifests)
            _validate_observed_msong_payload(identity_payload, region=args.region)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid --msong-manifest-root {manifests}: {exc}") from exc
        return manifests, identity
    inputs = work / "msong_inputs"
    manifests = inputs / "manifests"
    identity = manifests / "input_identity.json"
    if identity.is_file() and not args.force:
        missing = [
            name for name in ("calibration.json", "evaluation.json")
            if not (manifests / name).is_file()
        ]
        if missing:
            raise ValueError(
                f"incomplete cached MSONG manifests under {manifests}: {', '.join(missing)}"
            )
        _validate_msong_manifest_root(manifests)
        expected = _msong_index_contract(
            Path(args.msong_index_identity).expanduser().resolve(),
            args.msong_uri,
        )
        _validate_observed_managed_msong_index(args.msong_uri, expected)
        _, expected = _observe_msong_payload(
            args.msong_uri,
            expected,
            region=args.region,
        )
        observed = json.loads(identity.read_text(encoding="utf-8")).get("index_evidence")
        if observed != expected:
            raise ValueError(
                "cached MSONG manifests were traced from a different index identity; "
                "use --force or a new --out-dir"
            )
        return manifests, identity
    selected_npy = inputs / "selected_queries.npy"
    selection_json = inputs / "query_selection.json"
    trace_json = inputs / "query_ranges_trace.json"
    inputs.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable, str(HERE / "msong_select_queries.py"),
            "--queries", str(Path(args.msong_queries).expanduser().resolve()),
            "--output-npy", str(selected_npy), "--output-json", str(selection_json),
        ]
    )
    export_result = inputs / "export_result.json"
    env = dict(os.environ)
    env.update(
        {
            "AWS_DEFAULT_REGION": args.region,
            "LANCE_PAYLOAD_PLAN_ONLY": "1",
            "LANCE_PROCESS_IO_THREADS_LIMIT": "64",
        }
    )
    _run(
        [
            str(Path(args.trace_export_bin).expanduser().resolve()),
            "--uri", args.msong_uri, "--queries-npy", str(selected_npy),
            "--num-queries", "420", "--warmup-queries", "0",
            "--query-concurrency", "1", "--nprobe", "64", "--k", "10",
            "--refine-factor", "5", "--out-json", str(export_result),
            "--out-query-ranges-json", str(trace_json),
        ],
        env=env,
    )
    trace = json.loads(trace_json.read_text(encoding="utf-8"))
    uuids = sorted(set(map(str, trace.get("index_uuids", []))))
    contract = _msong_index_contract(
        Path(args.msong_index_identity).expanduser().resolve(),
        args.msong_uri,
    )
    _validate_observed_managed_msong_index(args.msong_uri, contract)
    if uuids != [contract["expected_index_uuid"]]:
        raise ValueError(
            "MSONG trace selected an index outside the declared identity: "
            f"observed={uuids}, expected={[contract['expected_index_uuid']]}"
        )
    object_key, contract = _observe_msong_payload(
        args.msong_uri,
        contract,
        region=args.region,
    )
    object_size = int(contract["observed_payload_object_size"])
    object_etag = str(contract["observed_payload_object_etag"])
    _run(
        [
            sys.executable, str(HERE / "msong_build_manifests.py"),
            "--trace", str(trace_json), "--selection", str(selection_json),
            "--dataset-uri", args.msong_uri, "--expected-index-uuid", uuids[0],
            "--object-size", str(object_size),
            "--object-etag", object_etag, "--output-root", str(manifests),
        ]
    )
    identity_payload = json.loads(identity.read_text(encoding="utf-8"))
    identity_payload["index_evidence"] = contract
    identity.write_text(
        json.dumps(identity_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifests, identity


def _msong(
    args: argparse.Namespace,
    work: Path,
    latency_model: Path,
    sift_policy: Path,
) -> Path:
    manifests, identity = _prepare_msong(args, work)
    calibration_dir = work / "msong_calibration"
    results_dir = work / "msong_results"
    policy = work / "msong_policies_before_evaluation.json"
    summary = work / "msong_summary_raw.json"
    calibration = _run_strict(
        args,
        label="calibration_msong_nprobe0064",
        dataset="msong",
        original=manifests / "calibration.json",
        reordered=manifests / "calibration.json",
        latency_model=latency_model,
        out_dir=calibration_dir,
        nprobe=64,
        gaps=GAPS,
        spans=SPANS,
        only_method="greedy_reorder",
    )
    _run([sys.executable, str(HERE / "msong_validate.py"), str(calibration), "--arm", "calibration"])
    if not policy.is_file() or args.force:
        _run(
            [
                sys.executable, str(HERE / "msong_analyze.py"), "freeze",
                "--calibration", str(calibration), "--sift-policy", str(sift_policy),
                "--output", str(policy),
            ]
        )
    frozen = json.loads(policy.read_text(encoding="utf-8"))
    msong_gap = int(frozen["msong_tuned_greedy"]["gap_kib"])
    msong_span = int(frozen["msong_tuned_greedy"]["span_cap_bytes"])
    orders = {
        1: ("sift_frozen_greedy", "msong_tuned_greedy", "warp_p32_k128"),
        2: ("msong_tuned_greedy", "warp_p32_k128", "sift_frozen_greedy"),
        3: ("warp_p32_k128", "sift_frozen_greedy", "msong_tuned_greedy"),
    }
    schedule_path = work / "msong_schedule.tsv"
    schedule_lines = ["repeat\torder\tarm\tfilename"]
    for repeat, arms in orders.items():
        for order, arm in enumerate(arms, start=1):
            label = f"repeat_{repeat}_order_{order}_{arm}"
            if arm == "sift_frozen_greedy":
                gap, span = 128, 2**20
            elif arm == "msong_tuned_greedy":
                gap, span = msong_gap, msong_span
            else:
                gap, span = 128, 2**20
            only_method = "warp_reorder" if arm == "warp_p32_k128" else "greedy_reorder"
            result = _run_strict(
                args,
                label=label,
                dataset="msong",
                original=manifests / "evaluation.json",
                reordered=manifests / "evaluation.json",
                latency_model=latency_model,
                out_dir=results_dir,
                nprobe=64,
                gaps=str(gap),
                spans=str(span // 2**20),
                only_method=only_method,
            )
            validate = [
                sys.executable, str(HERE / "msong_validate.py"), str(result), "--arm",
                "warp" if arm == "warp_p32_k128" else "greedy",
            ]
            if arm != "warp_p32_k128":
                validate.extend(["--gap-kib", str(gap), "--span-bytes", str(span)])
            _run(validate)
            schedule_lines.append(f"{repeat}\t{order}\t{arm}\t{label}.json")
    schedule_path.write_text("\n".join(schedule_lines) + "\n", encoding="utf-8")
    _run(
        [
            sys.executable, str(HERE / "msong_analyze.py"), "final",
            "--results-root", str(results_dir), "--schedule", str(schedule_path),
            "--freeze", str(policy), "--input-identity", str(identity),
            "--output", str(summary),
        ]
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    identity_payload = json.loads(identity.read_text(encoding="utf-8"))
    payload["index_evidence"] = identity_payload.get(
        "index_evidence",
        {
            "mode": "precomputed-manifests",
            "identity_path": str(identity),
            "identity_sha256": _sha256(identity),
        },
    )
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    strict, trace = _require_inputs(args)
    if args.check_only:
        print(
            f"[plan] strict_bin={strict} "
            f"only_method={'yes' if _binary_has_flag(strict, '--only-method') else 'no'}"
        )
        print(
            f"[plan] trace_export_bin={trace} "
            f"query_range_export={'yes' if _binary_has_flag(trace, '--out-query-ranges-json') else 'no'}"
        )
        print(f"[plan] bucket={args.bucket or '<missing>'} region={args.region}")
        print(f"[plan] msong_uri={args.msong_uri or '<missing>'}")
        print("[plan] SIFT=prepare + 54-cell calibration + 12 fixed-policy invocations")
        print("[plan] MSONG=trace + 54-cell calibration + 9 fixed-policy invocations")
        print("[plan] outputs=sift_summary.json,msong_summary.json")
        return 0

    out_dir = Path(args.out_dir).expanduser().resolve()
    work = out_dir / "work"
    contract = _measurement_contract(args, strict, trace)
    toolchain = contract["toolchain"]
    _bind_work_directory(work, contract, force=args.force)
    latency_model = HERE / "paper_latency_model.json"
    sift_summary, sift_policy = _sift(args, work, latency_model)
    msong_raw = _msong(args, work, latency_model, sift_policy)
    sift = json.loads(sift_summary.read_text(encoding="utf-8"))
    sift["reproduction_toolchain"] = toolchain
    (out_dir / "sift_summary.json").write_text(
        json.dumps(sift, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    msong = json.loads(msong_raw.read_text(encoding="utf-8"))
    msong["status"] = "complete"
    msong["policies"] = msong.pop("frozen_policies")
    msong["runtime"] = {
        "query_concurrency": MSONG.query_concurrency,
        "io_concurrency_limit": MSONG.io_limit,
        "nlist": 4096,
        "nprobe": 64,
        "repeats": MSONG.repeats,
        "measured_queries_per_repeat": MSONG.measured_queries,
        "warmup_queries_per_invocation": MSONG.warmup_queries,
    }
    msong["source_archive_sha256"] = msong.get("input_identity_sha256", "fresh-run")
    msong["source_summary_sha256"] = "fresh-run"
    msong["reproduction_toolchain"] = toolchain
    (out_dir / "msong_summary.json").write_text(
        json.dumps(msong, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] Figure 11 fresh evidence: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
QUERY_POOL_SCRIPT = HERE / "generate_lance_fixed_prefix_query_pool.py"
FANOUT_RUNNER_SCRIPT = HERE / "run_lance_1kb_fanout.py"

DEFAULT_S3_REGION = os.environ.get("VECIO_S3_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
DEFAULT_FILE_SIZE_BYTES = 512_000_000
DEFAULT_NLIST = 4096
DEFAULT_CLUSTER_BYTES = 1024
DEFAULT_MAX_FANOUT = 128
DEFAULT_FULL_NUM_QUERIES = 512
DEFAULT_FULL_WARMUP_QUERIES = 64
DEFAULT_SINGLE_NUM_QUERIES = 128
DEFAULT_SINGLE_WARMUP_QUERIES = 16
DEFAULT_SINGLE_PER_IO_NUM_QUERIES = 8192
DEFAULT_SINGLE_PER_IO_WARMUP_QUERIES = 128
DEFAULT_POOL_SIZE = 128
DEFAULT_QUERY_CONCURRENCY = 1
DEFAULT_SEED = 42

DATA_DIR = (
    ROOT
    / "experiments/figure-04-object-store-io/stages"
    / HERE.name
    / "data/latest"
)
GENERATED_DIR = DATA_DIR / "generated"
QUERYPOOL_Q512_DIR = GENERATED_DIR / "querypool_q512"
QUERYPOOL_Q128_DIR = GENERATED_DIR / "querypool_q128"
FANOUT_FULL_DIR = GENERATED_DIR / "fanout_full"
SINGLE_PER_IO_DIR = GENERATED_DIR / "single_per_iop"
SUMMARY_OUT = DATA_DIR / "summary.csv"
PER_IO_OUT = DATA_DIR / "fanout_001_per_iop.csv"
PROVENANCE_OUT = DATA_DIR / "data_provenance.json"

RUNTIME_OUT_ROOT = DATA_DIR
RUNTIME_GENERATED_DIR = GENERATED_DIR
RUNTIME_QUERYPOOL_Q512_DIR = QUERYPOOL_Q512_DIR
RUNTIME_QUERYPOOL_Q128_DIR = QUERYPOOL_Q128_DIR
RUNTIME_FANOUT_FULL_DIR = FANOUT_FULL_DIR
RUNTIME_SINGLE_PER_IO_DIR = SINGLE_PER_IO_DIR
RUNTIME_SUMMARY_OUT = SUMMARY_OUT
RUNTIME_PER_IO_OUT = PER_IO_OUT
RUNTIME_PROVENANCE_OUT = PROVENANCE_OUT


def _load_provenance_defaults() -> dict[str, str]:
    path = DATA_DIR / "data_provenance.json"
    if not path.exists():
        return {"uri": "", "bucket": "", "region": ""}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"uri": "", "bucket": "", "region": ""}

    exp = payload.get("experiment_parameters", payload if isinstance(payload, dict) else {})
    uri = str(exp.get("uri", "")).strip()
    bucket = ""
    region = ""
    if uri.startswith("s3://"):
        body = uri[5:]
        if "/" in body:
            bucket = body.split("/", 1)[0].strip()
    return {"uri": uri, "bucket": bucket, "region": ""}


PROVENANCE_DEFAULTS = _load_provenance_defaults()
DEFAULT_URI = os.environ.get("VECIO_FIG4_TINY_REQUEST_URI", PROVENANCE_DEFAULTS["uri"])
DEFAULT_S3_BUCKET = os.environ.get("VECIO_S3_BUCKET", PROVENANCE_DEFAULTS["bucket"])


def _set_runtime_out_root(path: Path) -> None:
    global RUNTIME_OUT_ROOT
    global RUNTIME_GENERATED_DIR
    global RUNTIME_QUERYPOOL_Q512_DIR
    global RUNTIME_QUERYPOOL_Q128_DIR
    global RUNTIME_FANOUT_FULL_DIR
    global RUNTIME_SINGLE_PER_IO_DIR
    global RUNTIME_SUMMARY_OUT
    global RUNTIME_PER_IO_OUT
    global RUNTIME_PROVENANCE_OUT
    RUNTIME_OUT_ROOT = path.resolve()
    RUNTIME_GENERATED_DIR = RUNTIME_OUT_ROOT / "generated"
    RUNTIME_QUERYPOOL_Q512_DIR = RUNTIME_GENERATED_DIR / "querypool_q512"
    RUNTIME_QUERYPOOL_Q128_DIR = RUNTIME_GENERATED_DIR / "querypool_q128"
    RUNTIME_FANOUT_FULL_DIR = RUNTIME_GENERATED_DIR / "fanout_full"
    RUNTIME_SINGLE_PER_IO_DIR = RUNTIME_GENERATED_DIR / "single_per_iop"
    RUNTIME_SUMMARY_OUT = RUNTIME_OUT_ROOT / "summary.csv"
    RUNTIME_PER_IO_OUT = RUNTIME_OUT_ROOT / "fanout_001_per_iop.csv"
    RUNTIME_PROVENANCE_OUT = RUNTIME_OUT_ROOT / "data_provenance.json"


def _upload_synthetic_blob(*, bucket: str, key: str, size_bytes: int, region: str) -> str:
    if size_bytes <= 0:
        raise SystemExit("--file-size-bytes must be > 0")

    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "boto3 is required for a live S3 run; install the full runner dependencies"
        ) from exc

    print(
        f"[upload] creating synthetic Figure 4 object s3://{bucket}/{key} ({int(size_bytes)} bytes)",
        flush=True,
    )
    s3 = boto3.client("s3", region_name=region)
    part_size = 8 * 1024 * 1024
    chunk = b"\0" * part_size
    upload_id = None
    parts: list[dict[str, object]] = []
    try:
        created = s3.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            ContentType="application/octet-stream",
            Metadata={"vecio-figure": "4", "vecio-generated": "true"},
        )
        upload_id = str(created["UploadId"])
        remaining = int(size_bytes)
        part_number = 1
        while remaining > 0:
            body = chunk if remaining >= part_size else chunk[:remaining]
            resp = s3.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=body,
            )
            parts.append({"ETag": resp["ETag"], "PartNumber": part_number})
            remaining -= len(body)
            part_number += 1
        s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        if upload_id is not None:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise

    head = s3.head_object(Bucket=bucket, Key=key)
    actual = int(head.get("ContentLength", -1))
    if actual != int(size_bytes):
        raise RuntimeError(
            f"synthetic Figure 4 object size mismatch for s3://{bucket}/{key}: expected {size_bytes}, got {actual}"
        )
    uri = f"s3://{bucket}/{key}"
    print(f"[upload] ready {uri}", flush=True)
    return uri


def _resolve_uri(args: argparse.Namespace) -> tuple[str, bool]:
    direct = str(args.uri).strip()
    if direct:
        return direct, False
    if str(args.s3_bucket).strip():
        bucket = str(args.s3_bucket).strip()
        key = str(args.s3_key).strip()
        if not key:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = f"figure04_synth/{ts}/blob_{int(args.file_size_bytes)}.bin"
        return _upload_synthetic_blob(
            bucket=bucket,
            key=key,
            size_bytes=int(args.file_size_bytes),
            region=str(args.s3_region).strip() or DEFAULT_S3_REGION,
        ), True
    repo_uri = str(PROVENANCE_DEFAULTS["uri"]).strip()
    if repo_uri:
        return repo_uri, False
    raise SystemExit("--uri or --s3-bucket is required unless the repo carries a default Figure 4 URI")


def _companion_binary(name: str) -> str | None:
    lance_root = str(os.environ.get("VECIO_LANCE_ROOT", "")).strip()
    if not lance_root:
        return None
    return str(
        Path(lance_root).expanduser().resolve()
        / "target"
        / "release"
        / "examples"
        / name
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Regenerate Figure 4 input data using repo scripts plus modified-Lance binaries."
    )
    ap.add_argument(
        "--lance-bin",
        default=_companion_binary("lance_scheduler_qps_s3"),
        help="Path to lance_scheduler_qps_s3.",
    )
    ap.add_argument(
        "--lance-per-iop-bin",
        default=_companion_binary("lance_scheduler_qps_s3_per_iop"),
        help="Path to lance_scheduler_qps_s3_per_iop.",
    )
    ap.add_argument("--uri", default=DEFAULT_URI, help="S3 URI for the merged cluster blob.")
    ap.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET, help="Bucket used when auto-generating the Figure 4 object.")
    ap.add_argument("--s3-region", default=DEFAULT_S3_REGION, help="Region used when auto-generating the Figure 4 object.")
    ap.add_argument("--s3-key", default="", help="Optional object key used when auto-generating the Figure 4 object.")
    ap.add_argument("--file-size-bytes", type=int, default=DEFAULT_FILE_SIZE_BYTES)
    ap.add_argument("--nlist", type=int, default=DEFAULT_NLIST)
    ap.add_argument("--cluster-bytes", type=int, default=DEFAULT_CLUSTER_BYTES)
    ap.add_argument("--max-fanout", type=int, default=DEFAULT_MAX_FANOUT)
    ap.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--query-concurrency", type=int, default=DEFAULT_QUERY_CONCURRENCY)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--full-num-queries", type=int, default=DEFAULT_FULL_NUM_QUERIES)
    ap.add_argument("--full-warmup-queries", type=int, default=DEFAULT_FULL_WARMUP_QUERIES)
    ap.add_argument("--single-num-queries", type=int, default=DEFAULT_SINGLE_NUM_QUERIES)
    ap.add_argument("--single-warmup-queries", type=int, default=DEFAULT_SINGLE_WARMUP_QUERIES)
    ap.add_argument("--single-per-iop-num-queries", type=int, default=DEFAULT_SINGLE_PER_IO_NUM_QUERIES)
    ap.add_argument("--single-per-iop-warmup-queries", type=int, default=DEFAULT_SINGLE_PER_IO_WARMUP_QUERIES)
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Print the commands and output plan without executing.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild generated/ before running.",
    )
    return ap.parse_args()


def _run(cmd: list[str], *, check_only: bool) -> None:
    print("$ " + " ".join(cmd))
    if check_only:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _ensure_clean_dir(path: Path, *, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_with_relative_paths(src: Path, dst: Path, *, relative_base: Path) -> None:
    with src.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    path_fields = {"txt_path", "per_query_csv", "per_io_csv"}
    for row in rows:
        for key in path_fields & set(row.keys()):
            raw = str(row.get(key, "") or "").strip()
            if not raw:
                continue
            resolved = relative_base / Path(raw).name
            row[key] = str(resolved.relative_to(RUNTIME_OUT_ROOT)) if resolved.is_relative_to(RUNTIME_OUT_ROOT) else str(resolved)

    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolved_s3_identity(args: argparse.Namespace) -> tuple[str, str]:
    uri = str(args.uri).strip()
    parsed = urlparse(uri)
    if parsed.scheme == "s3" and parsed.netloc:
        return parsed.netloc, parsed.path.lstrip("/")
    return str(args.s3_bucket).strip(), str(args.s3_key).strip()


def _write_metadata(args: argparse.Namespace) -> None:
    uri_arg = str(args.uri).strip() or "<s3://bucket/path/to/clusters_merged.bin>"
    s3_bucket, s3_key = _resolved_s3_identity(args)
    payload = {
        "figure": "fig04_real_s3_tiny_request_fanout_sanity",
        "uri": uri_arg,
        "s3_bucket": s3_bucket,
        "s3_region": str(args.s3_region).strip(),
        "s3_key": s3_key,
        "file_size_bytes": int(args.file_size_bytes),
        "nlist": int(args.nlist),
        "cluster_bytes": int(args.cluster_bytes),
        "max_fanout": int(args.max_fanout),
        "pool_size": int(args.pool_size),
        "query_concurrency": int(args.query_concurrency),
        "seed": int(args.seed),
        "generated_dirs": {
            "querypool_q512": str(RUNTIME_QUERYPOOL_Q512_DIR.relative_to(RUNTIME_OUT_ROOT)),
            "querypool_q128": str(RUNTIME_QUERYPOOL_Q128_DIR.relative_to(RUNTIME_OUT_ROOT)),
            "fanout_full": str(RUNTIME_FANOUT_FULL_DIR.relative_to(RUNTIME_OUT_ROOT)),
            "single_per_iop": str(RUNTIME_SINGLE_PER_IO_DIR.relative_to(RUNTIME_OUT_ROOT)),
        },
    }
    RUNTIME_PROVENANCE_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_root = Path(str(args.out_root)).expanduser().resolve() if str(args.out_root).strip() else DATA_DIR
    _set_runtime_out_root(out_root)
    uri_arg = str(args.uri).strip() or "<s3://bucket/path/to/clusters_merged.bin>"

    if int(args.query_concurrency) != 1:
        raise SystemExit("Figure 4 requires --query-concurrency 1")

    if not args.check_only:
        if not args.lance_bin:
            raise SystemExit("--lance-bin is required unless using --check-only")
        if not args.lance_per_iop_bin:
            raise SystemExit("--lance-per-iop-bin is required unless using --check-only")
        resolved_uri, generated = _resolve_uri(args)
        args.uri = resolved_uri
        uri_arg = resolved_uri
        print(
            f"[plan] figure4_uri={resolved_uri} "
            f"({'generated' if generated else 'configured'})",
            flush=True,
        )
    else:
        args.uri = uri_arg

    if not args.check_only:
        _ensure_clean_dir(RUNTIME_GENERATED_DIR, force=bool(args.force))
        for path in [
            RUNTIME_QUERYPOOL_Q512_DIR,
            RUNTIME_QUERYPOOL_Q128_DIR,
            RUNTIME_FANOUT_FULL_DIR,
            RUNTIME_SINGLE_PER_IO_DIR,
        ]:
            _ensure_clean_dir(path, force=False)

    querypool_q512_cmd = [
        sys.executable,
        str(QUERY_POOL_SCRIPT),
        "--uri",
        uri_arg,
        "--file-size-bytes",
        str(int(args.file_size_bytes)),
        "--out-dir",
        str(RUNTIME_QUERYPOOL_Q512_DIR),
        "--nlist",
        str(int(args.nlist)),
        "--cluster-bytes",
        str(int(args.cluster_bytes)),
        "--max-fanout",
        str(int(args.max_fanout)),
        "--num-queries",
        str(int(args.full_num_queries)),
        "--warmup-queries",
        str(int(args.full_warmup_queries)),
        "--seed",
        str(int(args.seed)),
    ]
    _run(querypool_q512_cmd, check_only=bool(args.check_only))

    fanout_full_cmd = [
        sys.executable,
        str(FANOUT_RUNNER_SCRIPT),
        "--bin",
        str(args.lance_bin or "<lance_scheduler_qps_s3>"),
        "--uri",
        uri_arg,
        "--fanouts",
        f"1-{int(args.max_fanout)}",
        "--nlist",
        str(int(args.nlist)),
        "--cluster-bytes",
        str(int(args.cluster_bytes)),
        "--pool-size",
        str(int(args.pool_size)),
        "--num-queries",
        str(int(args.full_num_queries)),
        "--warmup-queries",
        str(int(args.full_warmup_queries)),
        "--query-concurrency",
        str(int(args.query_concurrency)),
        "--query-ranges-dir",
        str(RUNTIME_QUERYPOOL_Q512_DIR),
        "--out-dir",
        str(RUNTIME_FANOUT_FULL_DIR),
    ]
    _run(fanout_full_cmd, check_only=bool(args.check_only))

    querypool_q128_cmd = [
        sys.executable,
        str(QUERY_POOL_SCRIPT),
        "--uri",
        uri_arg,
        "--file-size-bytes",
        str(int(args.file_size_bytes)),
        "--out-dir",
        str(RUNTIME_QUERYPOOL_Q128_DIR),
        "--nlist",
        str(int(args.nlist)),
        "--cluster-bytes",
        str(int(args.cluster_bytes)),
        "--max-fanout",
        str(int(args.max_fanout)),
        "--num-queries",
        str(int(args.single_num_queries)),
        "--warmup-queries",
        str(int(args.single_warmup_queries)),
        "--seed",
        str(int(args.seed)),
    ]
    _run(querypool_q128_cmd, check_only=bool(args.check_only))

    single_per_iop_cmd = [
        sys.executable,
        str(FANOUT_RUNNER_SCRIPT),
        "--bin",
        str(args.lance_per_iop_bin or "<lance_scheduler_qps_s3_per_iop>"),
        "--uri",
        uri_arg,
        "--fanouts",
        "1",
        "--nlist",
        str(int(args.nlist)),
        "--cluster-bytes",
        str(int(args.cluster_bytes)),
        "--pool-size",
        str(int(args.pool_size)),
        "--num-queries",
        str(int(args.single_per_iop_num_queries)),
        "--warmup-queries",
        str(int(args.single_per_iop_warmup_queries)),
        "--query-concurrency",
        str(int(args.query_concurrency)),
        "--record-per-iop",
        "--query-ranges-dir",
        str(RUNTIME_QUERYPOOL_Q128_DIR),
        "--out-dir",
        str(RUNTIME_SINGLE_PER_IO_DIR),
    ]
    _run(single_per_iop_cmd, check_only=bool(args.check_only))

    if args.check_only:
        print(f"[plan] canonical summary -> {RUNTIME_SUMMARY_OUT}")
        print(f"[plan] canonical per-iop -> {RUNTIME_PER_IO_OUT}")
        return 0

    generated_summary = RUNTIME_FANOUT_FULL_DIR / "summary.csv"
    generated_per_iop = RUNTIME_SINGLE_PER_IO_DIR / "fanout_001_per_iop.csv"
    if not generated_summary.exists():
        raise SystemExit(f"missing generated summary: {generated_summary}")
    if not generated_per_iop.exists():
        raise SystemExit(f"missing generated per-iop csv: {generated_per_iop}")

    _copy_with_relative_paths(
        generated_summary,
        RUNTIME_SUMMARY_OUT,
        relative_base=RUNTIME_FANOUT_FULL_DIR,
    )
    shutil.copy2(generated_per_iop, RUNTIME_PER_IO_OUT)
    _write_metadata(args)

    print(f"[ok] summary={RUNTIME_SUMMARY_OUT}")
    print(f"[ok] per_iop={RUNTIME_PER_IO_OUT}")
    print(f"[ok] provenance={RUNTIME_PROVENANCE_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

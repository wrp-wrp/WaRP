#!/usr/bin/env python3
"""
Probe S3 Range-GET latency vs bytes under realistic execution conditions.

Goals (SIGMOD hardening):
  - Warm up (connection reuse / steady state).
  - Measure serial per-request latency as a function of range size.
  - Measure concurrent behavior for k in [1..64] (homogeneous + mixed batches).
  - Fit and visualize w(bytes) shape (linear vs piecewise), plus residuals.

This script is intentionally independent from the pipeline stages to avoid
impacting ongoing experiments; it uses a dedicated S3 key prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

# Ensure project root is on sys.path when running as a standalone runner.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLOT_FORMATS = ("png", "pdf")
PAPER_DPI = 320
np: Any = None
plt: Any = None


def _load_probe_dependencies() -> None:
    """Load measurement/plotting dependencies only after CLI validation."""
    global np, plt
    try:
        import matplotlib
        import numpy as numpy_module

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "numpy and matplotlib are required to run the S3 calibration probe"
        ) from exc
    np = numpy_module
    plt = pyplot_module


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    for unit in ["KB", "MB", "GB"]:
        n_f = n / 1024.0
        if n_f < 1024.0:
            return f"{n_f:.1f}{unit}"
        n = int(n_f)
    return f"{n}B"


def _stable_size_tag(size_bytes: int) -> str:
    # Stable, filename-safe human tag for common powers-of-two units.
    if size_bytes % (1024 * 1024) == 0:
        return f"{size_bytes // (1024 * 1024)}mib"
    if size_bytes % 1024 == 0:
        return f"{size_bytes // 1024}kib"
    return f"{size_bytes}b"


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _savefig(out_dir: Path, stem: str, *, dpi: int = PAPER_DPI) -> List[str]:
    """
    Save the current matplotlib figure in all supported formats.

    Returns the relative plot filenames (e.g., ["foo.png", "foo.pdf"]).
    """
    saved: List[str] = []
    for ext in PLOT_FORMATS:
        p = out_dir / f"{stem}.{ext}"
        if ext == "png":
            plt.savefig(p, dpi=dpi)
        else:
            plt.savefig(p)
        saved.append(p.name)
    return saved


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_float(s: str) -> float:
    try:
        return float(str(s).strip())
    except Exception as e:
        raise ValueError(f"Invalid float: {s!r}") from e


def _default_sizes_bytes() -> List[int]:
    # Cover typical max_merge_span (10MB) plus headroom.
    return [
        1 * 1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
    ]


def _default_ks() -> List[int]:
    return [1, 2, 4, 8, 16, 32, 64]

def _parse_ranges(spec: str) -> List[Tuple[int, int, int]]:
    """
    Parse a mixed linear range spec like:
      "1:64:1,64:256:4,256:1024:16,1024:3072:64"
    meaning start_kib:end_kib:step_kib.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    ranges: List[Tuple[int, int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        items = [x.strip() for x in part.split(":")]
        if len(items) != 3:
            raise ValueError(f"Invalid range spec part: {part!r}. Expected start:end:step")
        a, b, c = (int(items[0]), int(items[1]), int(items[2]))
        if a <= 0 or b < a or c <= 0:
            raise ValueError(f"Invalid range values: {part!r}")
        ranges.append((a, b, c))
    return ranges


def _sizes_from_kib_ranges(ranges: List[Tuple[int, int, int]]) -> List[int]:
    sizes: List[int] = []
    for a, b, c in ranges:
        cur = a
        while cur <= b:
            sizes.append(cur * 1024)
            cur += c
    # Dedup while preserving order.
    out: List[int] = []
    seen = set()
    for s in sizes:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required when --config is used") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_s3_endpoint(
    *,
    config_path: str | None,
    cli_bucket: str | None,
    cli_region: str | None,
) -> Tuple[str, str]:
    """Resolve CLI > explicit YAML > environment without repo-local secrets."""
    config_s3: Dict[str, Any] = {}
    if config_path:
        config = _load_yaml(Path(config_path).expanduser())
        config_s3 = config.get("s3") or {}

    bucket = str(
        cli_bucket
        or config_s3.get("bucket")
        or os.environ.get("VECIO_S3_BUCKET", "")
    ).strip()
    region = str(
        cli_region
        or config_s3.get("region")
        or os.environ.get("VECIO_S3_REGION")
        or os.environ.get("AWS_DEFAULT_REGION", "")
    ).strip()

    if not bucket or not region:
        raise SystemExit(
            "Missing S3 endpoint: pass --s3-bucket/--s3-region, set "
            "VECIO_S3_BUCKET/VECIO_S3_REGION, or pass --config with s3.bucket/region"
        )
    placeholders = {
        "your-bucket",
        "your-bucket-name",
        "your-s3-bucket-name",
        "<bucket-name-from-user>",
    }
    if bucket.lower() in placeholders:
        raise SystemExit("S3 bucket is still a placeholder; provide a writable bucket")
    return bucket, region


def _make_s3_client(region: str, max_pool_connections: int) -> Any:
    try:
        import boto3
        import botocore.config
    except ModuleNotFoundError as exc:
        raise SystemExit("boto3 is required for a live S3 probe") from exc
    cfg = botocore.config.Config(max_pool_connections=max_pool_connections)
    return boto3.client("s3", region_name=region, config=cfg)


def _make_sts_client(region: str) -> Any:
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise SystemExit("boto3 is required for a live S3 probe") from exc
    return boto3.client("sts", region_name=region)


def _head_object(s3_client: Any, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _ensure_test_object(
    s3_client: Any,
    bucket: str,
    key: str,
    size_bytes: int,
    tmp_dir: Path,
) -> None:
    if _head_object(s3_client, bucket, key):
        return

    _mkdir(tmp_dir)
    local_path = tmp_dir / f"s3_model_probe_{size_bytes}.bin"

    # Create a file without allocating huge RAM.
    with local_path.open("wb") as f:
        f.truncate(size_bytes)

    s3_client.upload_file(str(local_path), bucket, key)


def _rand_offset(obj_size: int, length: int, rng: random.Random) -> int:
    if length <= 0:
        return 0
    if length >= obj_size:
        return 0
    return rng.randint(0, obj_size - length)


def _range_get_once(s3_client: Any, bucket: str, key: str, offset: int, length: int) -> float:
    # Return latency in ms.
    range_header = f"bytes={offset}-{offset + length - 1}"
    t0 = time.time()
    resp = s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
    _ = resp["Body"].read()
    return (time.time() - t0) * 1000.0


@dataclass
class SerialStats:
    size_bytes: int
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float


def _compute_stats(size_bytes: int, samples_ms: List[float]) -> SerialStats:
    arr = np.asarray(samples_ms, dtype=float)
    return SerialStats(
        size_bytes=size_bytes,
        n=int(arr.size),
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        std_ms=float(np.std(arr)),
    )


def _fit_linear(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    # Fit y = a + b*x
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    return a, b


def _fit_piecewise_two_segment(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Any]:
    # Brute force a single breakpoint among internal points.
    # Model: y = a1+b1*x for x<=t, y = a2+b2*x for x>t (not forced continuous).
    best = {
        "ok": False,
        "break_x": None,
        "a1": None,
        "b1": None,
        "a2": None,
        "b2": None,
        "sse": float("inf"),
    }
    n = xs.size
    if n < 6:
        return best

    for bi in range(2, n - 3):
        x1, y1 = xs[: bi + 1], ys[: bi + 1]
        x2, y2 = xs[bi + 1 :], ys[bi + 1 :]
        a1, b1 = _fit_linear(x1, y1)
        a2, b2 = _fit_linear(x2, y2)
        pred1 = a1 + b1 * x1
        pred2 = a2 + b2 * x2
        sse = float(np.sum((y1 - pred1) ** 2) + np.sum((y2 - pred2) ** 2))
        if sse < best["sse"]:
            best = {
                "ok": True,
                "break_x": float(xs[bi]),
                "a1": a1,
                "b1": b1,
                "a2": a2,
                "b2": b2,
                "sse": sse,
            }
    return best


def _predict_piecewise(xs: np.ndarray, piecewise: Dict[str, Any]) -> np.ndarray:
    if not piecewise.get("ok"):
        raise ValueError("piecewise fit not ok")
    bx = float(piecewise["break_x"])
    a1, b1 = float(piecewise["a1"]), float(piecewise["b1"])
    a2, b2 = float(piecewise["a2"]), float(piecewise["b2"])
    return np.where(xs <= bx, a1 + b1 * xs, a2 + b2 * xs)


def _error_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, Any]:
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    # Relative errors against observed y (avoid dividing by near-zero).
    denom = np.maximum(1e-9, y)
    rel = resid / denom
    return {
        "rmse_ms": rmse,
        "mae_ms": mae,
        "mape_percent": float(np.mean(np.abs(rel)) * 100.0),
        "max_abs_rel_percent": float(np.max(np.abs(rel)) * 100.0),
    }


def _plot_serial_curves(stats: List[SerialStats], out_dir: Path) -> None:
    xs = np.asarray([s.size_bytes for s in stats], dtype=float)
    p50 = np.asarray([s.p50_ms for s in stats], dtype=float)
    mean = np.asarray([s.mean_ms for s in stats], dtype=float)
    p95 = np.asarray([s.p95_ms for s in stats], dtype=float)
    p99 = np.asarray([s.p99_ms for s in stats], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, p50, marker="o", label="p50")
    plt.plot(xs, mean, marker="o", label="mean", alpha=0.85)
    plt.plot(xs, p95, marker="o", label="p95", alpha=0.85)
    plt.plot(xs, p99, marker="o", label="p99", alpha=0.85)
    plt.xscale("log")
    plt.xlabel("Range bytes (log)")
    plt.ylabel("Latency (ms)")
    plt.title("Serial Range-GET latency vs bytes")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _savefig(out_dir, "serial_latency_curves")
    plt.close()

    # Linear-x view in MiB for eyeballing straightness of affine models.
    plt.figure(figsize=(8, 5))
    xmb = xs / (1024.0 * 1024.0)
    plt.plot(xmb, p50, marker="o", label="p50")
    plt.plot(xmb, mean, marker="o", label="mean", alpha=0.85)
    plt.plot(xmb, p95, marker="o", label="p95", alpha=0.85)
    plt.plot(xmb, p99, marker="o", label="p99", alpha=0.85)
    plt.xlabel("Range size (MiB, linear)")
    plt.ylabel("Latency (ms)")
    plt.title("Serial Range-GET latency vs size (linear x)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _savefig(out_dir, "serial_latency_curves_linear_x")
    plt.close()

    # Zoomed view for 0..3MiB (paper-friendly / eyeballing small-range behavior).
    plt.figure(figsize=(8, 5))
    xmb = xs / (1024.0 * 1024.0)
    plt.plot(xmb, p50, marker="o", label="p50", markersize=3)
    plt.plot(xmb, mean, marker="o", label="mean", alpha=0.85, markersize=3)
    plt.plot(xmb, p95, marker="o", label="p95", alpha=0.85, markersize=3)
    plt.plot(xmb, p99, marker="o", label="p99", alpha=0.85, markersize=3)
    plt.xlim(0.0, 3.0)
    plt.xlabel("Range size (MiB, linear)")
    plt.ylabel("Latency (ms)")
    plt.title("Serial Range-GET latency vs size (0..3MiB zoom)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _savefig(out_dir, "serial_latency_curves_linear_x_zoom_0_3mib")
    plt.close()


def _plot_fit(xs: np.ndarray, ys: np.ndarray, a: float, b: float, piecewise: Dict[str, Any], out_dir: Path) -> None:
    pred = a + b * xs

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, label="measured (p50)", s=30)
    plt.plot(xs, pred, label=f"linear fit: O={a:.1f}ms, r={b:.3e}ms/byte")

    if piecewise.get("ok"):
        bx = float(piecewise["break_x"])
        a1, b1 = float(piecewise["a1"]), float(piecewise["b1"])
        a2, b2 = float(piecewise["a2"]), float(piecewise["b2"])
        pred_pw = np.where(xs <= bx, a1 + b1 * xs, a2 + b2 * xs)
        plt.plot(xs, pred_pw, label="piecewise (2-seg) fit", alpha=0.85)
        plt.axvline(bx, linestyle="--", alpha=0.5, label=f"break @ {_human_bytes(int(bx))}")

    plt.xscale("log")
    plt.xlabel("Range bytes (log)")
    plt.ylabel("Latency (ms)")
    plt.title("w(bytes) fit (p50)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _savefig(out_dir, "serial_p50_fit")
    plt.close()

    # Residuals
    rel = (ys - pred) / np.maximum(1e-9, ys)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(xs, rel * 100.0, s=30)
    plt.axhline(0.0, linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Range bytes (log)")
    plt.ylabel("Relative error (%)  (linear fit on p50)")
    plt.title("Fit residuals vs bytes")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _savefig(out_dir, "serial_p50_fit_residuals")
    plt.close()

    # Linear-x view (MiB) for direct "is it a line?" inspection.
    xmb = xs / (1024.0 * 1024.0)
    plt.figure(figsize=(8, 5))
    plt.scatter(xmb, ys, label="measured (p50)", s=30)
    plt.plot(xmb, pred, label=f"linear fit: O={a:.1f}ms, r={b:.3e}ms/byte")
    if piecewise.get("ok"):
        pred_pw = _predict_piecewise(xs, piecewise)
        bx = float(piecewise["break_x"]) / (1024.0 * 1024.0)
        plt.plot(xmb, pred_pw, label="piecewise (2-seg) fit", alpha=0.85)
        plt.axvline(bx, linestyle="--", alpha=0.5, label=f"break @ {bx:.2f}MiB")
    plt.xlabel("Range size (MiB, linear)")
    plt.ylabel("Latency (ms)")
    plt.title("w(bytes) fit (p50) (linear x)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _savefig(out_dir, "serial_p50_fit_linear_x")
    plt.close()

    relp = (ys - pred) / np.maximum(1e-9, ys) * 100.0
    plt.figure(figsize=(8, 4.5))
    plt.scatter(xmb, relp, s=30)
    plt.axhline(0.0, linestyle="--", alpha=0.5)
    plt.xlabel("Range size (MiB, linear)")
    plt.ylabel("Relative error (%)  (linear fit on p50)")
    plt.title("Fit residuals vs size (linear x)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _savefig(out_dir, "serial_p50_fit_residuals_linear_x")
    plt.close()

    # Zoomed fit view for 0..3MiB (linear x).
    xmb = xs / (1024.0 * 1024.0)
    plt.figure(figsize=(8, 5))
    plt.scatter(xmb, ys, label="measured (p50)", s=18)
    plt.plot(xmb, pred, label="linear fit")
    if piecewise.get("ok"):
        pred_pw = _predict_piecewise(xs, piecewise)
        plt.plot(xmb, pred_pw, label="piecewise (2-seg) fit", alpha=0.85)
    plt.xlim(0.0, 3.0)
    plt.xlabel("Range size (MiB, linear)")
    plt.ylabel("Latency (ms)")
    plt.title("w(bytes) fit (p50) (0..3MiB zoom)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _savefig(out_dir, "serial_p50_fit_linear_x_zoom_0_3mib")
    plt.close()


def _plot_concurrency(summary: Dict[str, Any], out_dir: Path) -> None:
    # Plot makespan vs k for each named scenario
    for name, rows in summary.items():
        ks = np.asarray([r["k"] for r in rows], dtype=float)
        p50 = np.asarray([r["p50_total_ms"] for r in rows], dtype=float)
        p95 = np.asarray([r["p95_total_ms"] for r in rows], dtype=float)
        mean = np.asarray([r["mean_total_ms"] for r in rows], dtype=float)

        plt.figure(figsize=(7.5, 4.8))
        plt.plot(ks, p50, marker="o", label="p50 makespan")
        plt.plot(ks, mean, marker="o", label="mean makespan", alpha=0.85)
        plt.plot(ks, p95, marker="o", label="p95 makespan", alpha=0.85)
        plt.xlabel("k (concurrent requests)")
        plt.ylabel("Batch wall time (ms)")
        plt.title(f"Concurrent batch makespan vs k ({name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        _savefig(out_dir, f"concurrent_makespan_{_slug(name)}")
        plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Optional YAML containing s3.bucket and s3.region")
    ap.add_argument("--s3-bucket", default=None, help="S3 bucket (otherwise config or VECIO_S3_BUCKET)")
    ap.add_argument("--s3-region", default=None, help="AWS region (otherwise config, VECIO_S3_REGION, or AWS_DEFAULT_REGION)")
    ap.add_argument("--prefix", default=None, help="S3 key prefix (default: network_calibration/model_probe/<ts>/)")
    ap.add_argument("--object-size-mb", type=int, default=256, help="Test object size in MB")
    ap.add_argument("--sizes-bytes", default=None, help="Comma-separated sizes in bytes (default: built-in list)")
    ap.add_argument("--sizes-linear-min-mib", default=None, help="Generate dense sizes: min MiB (linear spacing)")
    ap.add_argument("--sizes-linear-max-mib", default=None, help="Generate dense sizes: max MiB (linear spacing)")
    ap.add_argument("--sizes-linear-step-mib", default=None, help="Generate dense sizes: step MiB (linear spacing)")
    ap.add_argument("--sizes-kib-ranges", default=None, help="Mixed dense sizes in KiB: start:end:step,...")
    ap.add_argument("--serial-warmup", type=int, default=20, help="Warmup requests per size (serial)")
    ap.add_argument("--serial-samples", type=int, default=100, help="Measured requests per size (serial)")
    ap.add_argument("--serial-rounds", type=int, default=1, help="Repeat measurement rounds per size (stability)")
    ap.add_argument("--between-rounds-sleep-ms", type=int, default=200, help="Sleep between rounds (serial)")
    ap.add_argument("--concurrent-type", default="async_concurrent", help="concurrent|async_concurrent")
    ap.add_argument("--max-workers", type=int, default=64, help="Max workers for concurrent executors")
    ap.add_argument("--ks", default=None, help="Comma-separated k values (default: 1,2,4,8,16,32,64)")
    ap.add_argument("--conc-warmup-batches", type=int, default=3, help="Warmup batches per k")
    ap.add_argument("--conc-batches", type=int, default=8, help="Measured batches per k")
    ap.add_argument("--conc-sizes-bytes", default=None, help="Comma-separated sizes in bytes for the concurrency phase (default: legacy representative sizes)")
    ap.add_argument("--skip-concurrency", action="store_true", help="Only run serial measurements")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="Output directory (default: results/model_probe/<ts>/)")
    args = ap.parse_args()
    t_all0 = time.time()

    def _log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    bucket, region = _resolve_s3_endpoint(
        config_path=args.config,
        cli_bucket=args.s3_bucket,
        cli_region=args.s3_region,
    )
    _load_probe_dependencies()

    tag = _now_tag()
    prefix = args.prefix or f"network_calibration/model_probe/{tag}"
    key = f"{prefix}/blob_{args.object_size_mb}mb.bin"

    out_dir = Path(args.out) if args.out else (Path("results") / "model_probe" / f"{tag}_{region}_{args.concurrent_type}_k64")
    plots_dir = out_dir / "plots"
    tmp_dir = out_dir / "tmp"
    _mkdir(plots_dir)
    _mkdir(tmp_dir)

    rng = random.Random(args.seed)

    if args.sizes_kib_ranges:
        ranges = _parse_ranges(args.sizes_kib_ranges)
        sizes = _sizes_from_kib_ranges(ranges)
    elif args.sizes_linear_min_mib is not None or args.sizes_linear_max_mib is not None or args.sizes_linear_step_mib is not None:
        if args.sizes_linear_min_mib is None or args.sizes_linear_max_mib is None or args.sizes_linear_step_mib is None:
            raise SystemExit("--sizes-linear-* requires min/max/step all set")
        mn = _parse_float(args.sizes_linear_min_mib)
        mx = _parse_float(args.sizes_linear_max_mib)
        st = _parse_float(args.sizes_linear_step_mib)
        if not (mn > 0 and mx >= mn and st > 0):
            raise SystemExit("Invalid --sizes-linear-* values")
        sizes = []
        cur = mn
        while cur <= mx + 1e-9:
            sizes.append(int(round(cur * 1024.0 * 1024.0)))
            cur += st
    else:
        sizes = _default_sizes_bytes() if not args.sizes_bytes else _parse_int_list(args.sizes_bytes)

    ks = _default_ks() if not args.ks else _parse_int_list(args.ks)
    ks = [k for k in ks if 1 <= k <= 1024]
    ks = sorted(set(ks))

    meta: Dict[str, Any] = {
        "time": tag,
        "bucket": bucket,
        "region": region,
        "s3_key": key,
        "object_size_mb": int(args.object_size_mb),
        "plot_formats": list(PLOT_FORMATS),
        "paper_dpi": int(PAPER_DPI),
        "sizes_bytes": sizes,
        "serial_warmup": int(args.serial_warmup),
        "serial_samples": int(args.serial_samples),
        "concurrent_type": args.concurrent_type,
        "max_workers": int(args.max_workers),
        "ks": ks,
        "conc_warmup_batches": int(args.conc_warmup_batches),
        "conc_batches": int(args.conc_batches),
        "seed": int(args.seed),
    }
    _log(
        "Probe start: "
        f"bucket={bucket}, region={region}, object={args.object_size_mb}MiB, "
        f"sizes={len(sizes)}, serial_samples={int(args.serial_samples)}, "
        f"serial_rounds={int(args.serial_rounds)}, skip_concurrency={bool(args.skip_concurrency)}"
    )
    _log(f"Output dir: {out_dir}")

    # Connection pool sizing: leave headroom above max_workers.
    pool = max(32, int(args.max_workers) * 2)
    s3_client = _make_s3_client(region, max_pool_connections=pool)

    # Ensure credentials early.
    try:
        _make_sts_client(region).get_caller_identity()
    except Exception as e:
        raise SystemExit(f"AWS credentials not configured or not permitted: {e}")
    _log("AWS credentials check: OK")

    # Prepare test object (dedicated key prefix).
    obj_size = int(args.object_size_mb) * 1024 * 1024
    _ensure_test_object(s3_client, bucket, key, obj_size, tmp_dir)
    _log(f"Test object ready: s3://{bucket}/{key}")

    # ----- Phase 1: serial measurement (single-request) -----
    t_serial0 = time.time()
    serial_raw: Dict[int, List[float]] = {}
    serial_rounds_raw: Dict[int, List[List[float]]] = {}
    valid_sizes = [int(sz) for sz in sizes if int(sz) > 0 and int(sz) <= obj_size]
    for sidx, size in enumerate(valid_sizes, start=1):
        if size <= 0 or size > obj_size:
            continue
        t_size0 = time.time()
        _log(
            f"[serial {sidx}/{len(valid_sizes)}] size={_human_bytes(size)} "
            f"(warmup={int(args.serial_warmup)}, samples={int(args.serial_samples)}, rounds={max(1, int(args.serial_rounds))})"
        )
        # Warmup
        for _ in range(int(args.serial_warmup)):
            off = _rand_offset(obj_size, size, rng)
            _ = _range_get_once(s3_client, bucket, key, off, size)

        rounds = max(1, int(args.serial_rounds))
        per_round: List[List[float]] = []
        for ridx in range(rounds):
            samples: List[float] = []
            for _ in range(int(args.serial_samples)):
                off = _rand_offset(obj_size, size, rng)
                lat = _range_get_once(s3_client, bucket, key, off, size)
                samples.append(float(lat))
            per_round.append(samples)
            arr = np.asarray(samples, dtype=float)
            _log(
                f"  round {ridx + 1}/{rounds}: "
                f"p50={float(np.percentile(arr, 50)):.2f}ms "
                f"p95={float(np.percentile(arr, 95)):.2f}ms "
                f"mean={float(np.mean(arr)):.2f}ms"
            )
            if ridx + 1 < rounds:
                time.sleep(max(0.0, float(args.between_rounds_sleep_ms)) / 1000.0)
        serial_rounds_raw[size] = per_round
        serial_raw[size] = [x for rr in per_round for x in rr]
        _log(f"  done in {time.time() - t_size0:.1f}s")

    with (out_dir / "serial_rounds_raw.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in serial_rounds_raw.items()}, f)

    serial_stats = [_compute_stats(sz, lats) for sz, lats in sorted(serial_raw.items(), key=lambda x: x[0])]
    with (out_dir / "serial_stats.json").open("w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in serial_stats], f, indent=2)
    _log(f"Serial phase complete in {time.time() - t_serial0:.1f}s")

    # Fit on p50 to reduce tail pollution.
    xs = np.asarray([s.size_bytes for s in serial_stats], dtype=float)
    ys = np.asarray([s.p50_ms for s in serial_stats], dtype=float)
    a, b = _fit_linear(xs, ys)
    piecewise = _fit_piecewise_two_segment(xs, ys)

    # Evaluate fits on the observed points.
    yhat_lin = a + b * xs
    metrics_lin = _error_metrics(ys, yhat_lin)

    metrics_pw = None
    yhat_pw = None
    if piecewise.get("ok"):
        yhat_pw = _predict_piecewise(xs, piecewise)
        metrics_pw = _error_metrics(ys, yhat_pw)

    # Simple holdout check (even/odd indices) to guard against overfitting.
    idx = np.arange(xs.size)
    tr = idx[idx % 2 == 0]
    te = idx[idx % 2 == 1]
    a2, b2 = _fit_linear(xs[tr], ys[tr])
    hold_lin = _error_metrics(ys[te], a2 + b2 * xs[te])

    hold_pw = None
    if piecewise.get("ok") and xs.size >= 10:
        # Fit piecewise by searching break on train set only.
        pw_tr = _fit_piecewise_two_segment(xs[tr], ys[tr])
        if pw_tr.get("ok"):
            hold_pw = _error_metrics(ys[te], _predict_piecewise(xs[te], pw_tr))

    fit_report: Dict[str, Any] = {
        "linear": {"O_ms": a, "r_ms_per_byte": b, "metrics": metrics_lin, "holdout_metrics": hold_lin},
        "piecewise_2seg": {
            **piecewise,
            **({"metrics": metrics_pw} if metrics_pw is not None else {}),
            **({"holdout_metrics": hold_pw} if hold_pw is not None else {}),
        },
    }
    with (out_dir / "fit_report.json").open("w", encoding="utf-8") as f:
        json.dump(fit_report, f, indent=2)

    # Suggested cost_model snippet for plugging into pipeline configs/results.
    cost_model_suggestion: Dict[str, Any] = {
        "linear": {
            "latency_model": "linear",
            "latency_per_request": float(a),
            "latency_per_byte": float(b),
        },
    }
    if piecewise.get("ok"):
        cost_model_suggestion["piecewise_linear"] = {
            "latency_model": "piecewise_linear",
            "latency_pw_breaks_bytes": [int(piecewise["break_x"])],
            "latency_pw_overheads_ms": [float(piecewise["a1"]), float(piecewise["a2"])],
            "latency_pw_rates_ms_per_byte": [float(piecewise["b1"]), float(piecewise["b2"])],
        }
    with (out_dir / "cost_model_suggestion.json").open("w", encoding="utf-8") as f:
        json.dump(cost_model_suggestion, f, indent=2)

    _plot_serial_curves(serial_stats, plots_dir)
    _plot_fit(xs, ys, a, b, piecewise, plots_dir)
    _log("Serial plots + fit artifacts written.")

    # ----- Phase 2: concurrent measurement (batch makespan) -----
    if args.skip_concurrency:
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        report = []
        report.append("# S3 Range Cost Model Probe (serial-only)")
        report.append("")
        report.append(f"- bucket: `{bucket}`")
        report.append(f"- region: `{region}`")
        report.append(f"- s3_key: `{key}`")
        report.append(f"- object_size: `{args.object_size_mb}MB`")
        report.append(f"- serial sizes: `{', '.join(_human_bytes(s) for s in sizes)}`")
        report.append("")
        report.append("## Linear Fit (p50)")
        report.append(f"- O (latency_per_request): `{a:.2f} ms`")
        report.append(f"- r (latency_per_byte): `{b:.6e} ms/byte`")
        report.append(f"- fit error: max_abs_rel=`{metrics_lin['max_abs_rel_percent']:.2f}%`, mape=`{metrics_lin['mape_percent']:.2f}%`, rmse=`{metrics_lin['rmse_ms']:.2f}ms`")
        report.append(f"- holdout error: max_abs_rel=`{hold_lin['max_abs_rel_percent']:.2f}%`, mape=`{hold_lin['mape_percent']:.2f}%`, rmse=`{hold_lin['rmse_ms']:.2f}ms`")
        if piecewise.get('ok') and metrics_pw is not None:
            report.append("")
            report.append("## Piecewise Fit (2 segments, p50)")
            report.append(f"- break @ `{_human_bytes(int(piecewise['break_x']))}`")
            report.append(f"- fit error: max_abs_rel=`{metrics_pw['max_abs_rel_percent']:.2f}%`, mape=`{metrics_pw['mape_percent']:.2f}%`, rmse=`{metrics_pw['rmse_ms']:.2f}ms`")
            if hold_pw is not None:
                report.append(f"- holdout error: max_abs_rel=`{hold_pw['max_abs_rel_percent']:.2f}%`, mape=`{hold_pw['mape_percent']:.2f}%`, rmse=`{hold_pw['rmse_ms']:.2f}ms`")
        report.append("")
        report.append("## Plots")
        report.append("- All plots are saved as `{png,pdf}`.")
        report.append("- `plots/serial_latency_curves.{png,pdf}`")
        report.append("- `plots/serial_latency_curves_linear_x.{png,pdf}`")
        report.append("- `plots/serial_latency_curves_linear_x_zoom_0_3mib.{png,pdf}`")
        report.append("- `plots/serial_p50_fit.{png,pdf}`")
        report.append("- `plots/serial_p50_fit_residuals.{png,pdf}`")
        report.append("- `plots/serial_p50_fit_linear_x.{png,pdf}`")
        report.append("- `plots/serial_p50_fit_residuals_linear_x.{png,pdf}`")
        report.append("- `plots/serial_p50_fit_linear_x_zoom_0_3mib.{png,pdf}`")
        report.append("")
        report.append("## Suggested Config Snippet")
        report.append("- `cost_model_suggestion.json`")
        report.append("")
        with (out_dir / "REPORT.md").open("w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
        _log(f"Probe done (serial-only) in {time.time() - t_all0:.1f}s")
        return 0

    try:
        from src.request_executor import create_executor
    except ModuleNotFoundError as exc:
        raise SystemExit("boto3 is required for a live S3 probe") from exc
    conc_executor = create_executor(
        args.concurrent_type,
        max_workers=int(args.max_workers),
        region=region,
        pool_size=max(1, min(16, int(args.max_workers) // 4)),
    )

    # Concurrency sizes: explicit override, else legacy representative sizes.
    if args.conc_sizes_bytes:
        conc_sizes = [s for s in _parse_int_list(args.conc_sizes_bytes) if 0 < s <= obj_size]
        if not conc_sizes:
            raise SystemExit("No valid sizes in --conc-sizes-bytes (check --object-size-mb)")
    else:
        conc_sizes = []
        for cand in [16 * 1024, 1 * 1024 * 1024, 8 * 1024 * 1024]:
            if cand <= obj_size:
                conc_sizes.append(cand)
        if not conc_sizes:
            valid = [int(s) for s in sizes if 0 < int(s) <= obj_size]
            if not valid:
                raise SystemExit("No valid sizes for concurrency (check --object-size-mb / --sizes-*)")
            conc_sizes = [min(valid), max(valid)]

    def run_batches(name: str, make_plan_fn) -> List[Dict[str, Any]]:
        rows = []
        _log(
            f"[concurrency] scenario={name}, ks={ks}, "
            f"warmup_batches={int(args.conc_warmup_batches)}, measured_batches={int(args.conc_batches)}"
        )
        for k in ks:
            # Warmup batches
            for _ in range(int(args.conc_warmup_batches)):
                plan = make_plan_fn(k)
                _ = conc_executor.execute(plan, None, bucket, key)
            # Measure batches
            totals: List[float] = []
            per_req_all: List[float] = []
            for _ in range(int(args.conc_batches)):
                plan = make_plan_fn(k)
                res = conc_executor.execute(plan, None, bucket, key)
                totals.append(float(res.total_latency_ms))
                per_req_all.extend([float(x) for x in res.per_request_latencies])
            tot_arr = np.asarray(totals, dtype=float)
            pr_arr = np.asarray(per_req_all, dtype=float) if per_req_all else np.asarray([], dtype=float)
            row = {
                "k": int(k),
                "batches": int(len(totals)),
                "mean_total_ms": float(np.mean(tot_arr)),
                "p50_total_ms": float(np.percentile(tot_arr, 50)),
                "p95_total_ms": float(np.percentile(tot_arr, 95)),
                "mean_per_request_ms": float(np.mean(pr_arr)) if pr_arr.size else None,
                "p50_per_request_ms": float(np.percentile(pr_arr, 50)) if pr_arr.size else None,
                "p95_per_request_ms": float(np.percentile(pr_arr, 95)) if pr_arr.size else None,
            }
            rows.append(row)
            _log(f"  k={int(k)}: total p50={row['p50_total_ms']:.2f}ms p95={row['p95_total_ms']:.2f}ms")
        return rows

    conc_summary: Dict[str, Any] = {}
    for sz in conc_sizes:
        def make_plan_hom(k: int, size=sz):
            plan = []
            for _ in range(k):
                off = _rand_offset(obj_size, size, rng)
                plan.append((off, int(size)))
            return plan
        scenario = f"hom_{_stable_size_tag(int(sz))}"
        conc_summary[scenario] = run_batches(scenario, make_plan_hom)

    # Mixed: sample sizes from the serial size set (biased toward typical region).
    weights = np.asarray([1.0 / math.sqrt(max(1.0, float(s))) for s in sizes], dtype=float)
    weights = weights / np.sum(weights)

    def make_plan_mixed(k: int):
        plan = []
        for _ in range(k):
            size = int(rng.choices(sizes, weights=weights.tolist(), k=1)[0])
            off = _rand_offset(obj_size, size, rng)
            plan.append((off, size))
        return plan

    conc_summary["mixed_weighted"] = run_batches("mixed", make_plan_mixed)

    with (out_dir / "concurrent_summary.json").open("w", encoding="utf-8") as f:
        json.dump(conc_summary, f, indent=2)

    _plot_concurrency(conc_summary, plots_dir)

    conc_executor.close()

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Lightweight human-readable report for quick inspection.
    report = []
    report.append("# S3 Range Cost Model Probe")
    report.append("")
    report.append(f"- bucket: `{bucket}`")
    report.append(f"- region: `{region}`")
    report.append(f"- s3_key: `{key}`")
    report.append(f"- object_size: `{args.object_size_mb}MB`")
    report.append(f"- serial sizes: `{', '.join(_human_bytes(s) for s in sizes)}`")
    report.append(f"- concurrent ks: `{ks}` (type={args.concurrent_type}, max_workers={args.max_workers})")
    report.append("")
    report.append("## Linear Fit (p50)")
    report.append(f"- O (latency_per_request): `{a:.2f} ms`")
    report.append(f"- r (latency_per_byte): `{b:.6e} ms/byte`")
    report.append(f"- fit error: max_abs_rel=`{metrics_lin['max_abs_rel_percent']:.2f}%`, mape=`{metrics_lin['mape_percent']:.2f}%`, rmse=`{metrics_lin['rmse_ms']:.2f}ms`")
    report.append(f"- holdout error: max_abs_rel=`{hold_lin['max_abs_rel_percent']:.2f}%`, mape=`{hold_lin['mape_percent']:.2f}%`, rmse=`{hold_lin['rmse_ms']:.2f}ms`")
    if piecewise.get("ok"):
        report.append("")
        report.append("## Piecewise Fit (2 segments, p50)")
        report.append(f"- break @ `{_human_bytes(int(piecewise['break_x']))}`")
        report.append(f"- seg1: O=`{piecewise['a1']:.2f}`, r=`{piecewise['b1']:.6e}`")
        report.append(f"- seg2: O=`{piecewise['a2']:.2f}`, r=`{piecewise['b2']:.6e}`")
        if metrics_pw is not None:
            report.append(f"- fit error: max_abs_rel=`{metrics_pw['max_abs_rel_percent']:.2f}%`, mape=`{metrics_pw['mape_percent']:.2f}%`, rmse=`{metrics_pw['rmse_ms']:.2f}ms`")
        if hold_pw is not None:
            report.append(f"- holdout error: max_abs_rel=`{hold_pw['max_abs_rel_percent']:.2f}%`, mape=`{hold_pw['mape_percent']:.2f}%`, rmse=`{hold_pw['rmse_ms']:.2f}ms`")
    report.append("")
    report.append("## Plots")
    report.append("- All plots are saved as `{png,pdf}`.")
    report.append("- `plots/serial_latency_curves.{png,pdf}`")
    report.append("- `plots/serial_latency_curves_linear_x.{png,pdf}`")
    report.append("- `plots/serial_latency_curves_linear_x_zoom_0_3mib.{png,pdf}`")
    report.append("- `plots/serial_p50_fit.{png,pdf}`")
    report.append("- `plots/serial_p50_fit_residuals.{png,pdf}`")
    report.append("- `plots/serial_p50_fit_linear_x.{png,pdf}`")
    report.append("- `plots/serial_p50_fit_residuals_linear_x.{png,pdf}`")
    report.append("- `plots/serial_p50_fit_linear_x_zoom_0_3mib.{png,pdf}`")
    report.append("")
    report.append("## Suggested Config Snippet")
    report.append("- `cost_model_suggestion.json`")
    for k in conc_summary.keys():
        report.append(f"- `plots/concurrent_makespan_{_slug(k)}.{{png,pdf}}`")
    report.append("")

    with (out_dir / "REPORT.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    _log(f"Probe done (serial+concurrency) in {time.time() - t_all0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

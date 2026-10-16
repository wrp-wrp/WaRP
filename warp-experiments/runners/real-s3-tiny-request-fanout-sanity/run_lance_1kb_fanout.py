#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANCE_ROOT = ROOT.parent / "lance"
DEFAULT_RELEASE_BIN = DEFAULT_LANCE_ROOT / "target" / "release" / "examples" / "lance_scheduler_qps_s3"
DEFAULT_DEBUG_BIN = DEFAULT_LANCE_ROOT / "target" / "debug" / "examples" / "lance_scheduler_qps_s3"


@dataclass(frozen=True)
class RunMetrics:
    fanout: int
    input_ranges: float
    planned_ranges_avg: float
    planned_ranges_min: float
    planned_ranges_max: float
    payload_bytes_avg: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    wall_ms: float
    qps: float
    measured_requests: int
    measured_iops: int
    measured_bytes: int
    txt_path: Path
    per_query_csv: Path
    per_io_csv: Path | None


def _default_bin() -> Path:
    if DEFAULT_RELEASE_BIN.exists():
        return DEFAULT_RELEASE_BIN
    if DEFAULT_DEBUG_BIN.exists():
        return DEFAULT_DEBUG_BIN
    return DEFAULT_RELEASE_BIN


def _parse_int_spec(raw: str) -> list[int]:
    out: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"invalid range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    vals = sorted(v for v in out if v > 0)
    if not vals:
        raise ValueError("empty fanout set")
    return vals


def _extract_float(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"failed to parse float with pattern: {pattern}")
    return float(m.group(1))


def _extract_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"failed to parse int with pattern: {pattern}")
    return int(m.group(1))


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _load_per_query(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "input_ranges": float(row["input_ranges"]),
                    "latency_ms": float(row["latency_ms"]),
                    "payload_bytes": float(row["payload_bytes"]),
                    "planned_ranges": float(row["planned_ranges"]),
                }
            )
    if not rows:
        raise RuntimeError(f"no per-query rows in {csv_path}")
    return rows


def _run_one(
    *,
    bin_path: Path,
    uri: str,
    out_dir: Path,
    fanout: int,
    nlist: int,
    cluster_bytes: int,
    pool_size: int,
    num_queries: int,
    warmup_queries: int,
    query_concurrency: int,
    seed: int,
    preplan: bool,
    max_iop_size: int,
    query_ranges_file: Path | None,
    record_per_iop: bool,
) -> RunMetrics:
    txt_path = out_dir / f"fanout_{fanout:03d}.txt"
    per_query_csv = out_dir / f"fanout_{fanout:03d}_per_query.csv"
    per_io_csv = out_dir / f"fanout_{fanout:03d}_per_iop.csv" if record_per_iop else None
    result_kv_path = out_dir / f"fanout_{fanout:03d}_result.txt"

    env = dict(os.environ)
    env.update(
        {
            "LANCE_QPS_URI": uri,
            "LANCE_QPS_NLIST": str(int(nlist)),
            "LANCE_QPS_NPROBE": str(int(fanout)),
            "LANCE_QPS_NUM_QUERIES": str(int(num_queries)),
            "LANCE_QPS_WARMUP_QUERIES": str(int(warmup_queries)),
            "LANCE_QPS_QUERY_CONCURRENCY": str(int(query_concurrency)),
            "LANCE_QPS_SEED": str(int(seed)),
            "LANCE_QPS_RECORD_PER_QUERY": "true",
            "LANCE_QPS_OUT_PER_QUERY": str(per_query_csv),
            "LANCE_QPS_OUT_JSON": str(result_kv_path),
            "LANCE_QPS_PREPLAN": "true" if preplan else "false",
            "LANCE_PROCESS_IO_THREADS_LIMIT": str(int(pool_size)),
            "LANCE_MAX_IOP_SIZE": str(int(max_iop_size)),
            "LANCE_IO_MERGE_STRATEGY": "separate",
            "LANCE_IO_MERGE_LATENCY_STD_MS": "0",
        }
    )
    if per_io_csv is not None:
        env["LANCE_QPS_OUT_PER_IO"] = str(per_io_csv)
    if query_ranges_file is not None:
        env["LANCE_QPS_QUERY_RANGES_FILE"] = str(query_ranges_file)
    else:
        env["LANCE_QPS_SYNTH_CLUSTER_BYTES"] = str(int(cluster_bytes))

    with txt_path.open("w", encoding="utf-8") as f:
        subprocess.run([str(bin_path)], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    result_text = result_kv_path.read_text(encoding="utf-8", errors="ignore")
    rows = _load_per_query(per_query_csv)
    latencies = sorted(float(r["latency_ms"]) for r in rows)
    input_ranges = [float(r["input_ranges"]) for r in rows]
    planned_ranges = [float(r["planned_ranges"]) for r in rows]
    payload_bytes = [float(r["payload_bytes"]) for r in rows]

    return RunMetrics(
        fanout=int(fanout),
        input_ranges=statistics.mean(input_ranges),
        planned_ranges_avg=statistics.mean(planned_ranges),
        planned_ranges_min=min(planned_ranges),
        planned_ranges_max=max(planned_ranges),
        payload_bytes_avg=statistics.mean(payload_bytes),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        p99_latency_ms=_percentile(latencies, 0.99),
        max_latency_ms=max(latencies),
        wall_ms=_extract_float(result_text, r"wall_ms=([0-9]+(?:\.[0-9]+)?)"),
        qps=_extract_float(result_text, r"qps=([0-9]+(?:\.[0-9]+)?)"),
        measured_requests=_extract_int(result_text, r"measured_requests=(\d+)"),
        measured_iops=_extract_int(result_text, r"measured_iops=(\d+)"),
        measured_bytes=_extract_int(result_text, r"measured_bytes=(\d+)"),
        txt_path=txt_path,
        per_query_csv=per_query_csv,
        per_io_csv=per_io_csv,
    )


def _write_csv(rows: Iterable[RunMetrics], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fanout",
                "input_ranges_avg",
                "planned_ranges_avg",
                "planned_ranges_min",
                "planned_ranges_max",
                "payload_bytes_avg",
                "avg_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "max_latency_ms",
                "wall_ms",
                "qps",
                "measured_requests",
                "measured_iops",
                "measured_bytes",
                "txt_path",
                "per_query_csv",
                "per_io_csv",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.fanout,
                    f"{row.input_ranges:.6f}",
                    f"{row.planned_ranges_avg:.6f}",
                    f"{row.planned_ranges_min:.6f}",
                    f"{row.planned_ranges_max:.6f}",
                    f"{row.payload_bytes_avg:.6f}",
                    f"{row.avg_latency_ms:.6f}",
                    f"{row.p50_latency_ms:.6f}",
                    f"{row.p95_latency_ms:.6f}",
                    f"{row.p99_latency_ms:.6f}",
                    f"{row.max_latency_ms:.6f}",
                    f"{row.wall_ms:.6f}",
                    f"{row.qps:.6f}",
                    row.measured_requests,
                    row.measured_iops,
                    row.measured_bytes,
                    str(row.txt_path),
                    str(row.per_query_csv),
                    str(row.per_io_csv) if row.per_io_csv is not None else "",
                ]
            )


def _write_readme(
    *,
    out_path: Path,
    args: argparse.Namespace,
    rows: list[RunMetrics],
    bin_path: Path,
) -> None:
    lines = [
        "Lance 1KB fanout sweep",
        "",
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"bin={bin_path}",
        f"uri={args.uri}",
        f"fanouts={','.join(str(x) for x in args.fanouts)}",
        f"nlist={args.nlist}",
        f"cluster_bytes={args.cluster_bytes}",
        f"pool_size={args.pool_size}",
        f"num_queries={args.num_queries}",
        f"warmup_queries={args.warmup_queries}",
        f"query_concurrency={args.query_concurrency}",
        f"preplan={args.preplan}",
        f"record_per_iop={args.record_per_iop}",
        "",
        "summary_columns=fanout,avg_latency_ms,p95_latency_ms,max_latency_ms,planned_ranges_avg",
    ]
    for row in rows:
        lines.append(
            f"{row.fanout},{row.avg_latency_ms:.3f},{row.p95_latency_ms:.3f},{row.max_latency_ms:.3f},{row.planned_ranges_avg:.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_plot(rows: list[RunMetrics], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fanouts = [r.fanout for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(fanouts, [r.avg_latency_ms for r in rows], marker="o", linewidth=1.8, label="avg")
    ax.plot(fanouts, [r.p95_latency_ms for r in rows], marker="s", linewidth=1.6, label="p95")
    ax.plot(fanouts, [r.max_latency_ms for r in rows], marker="^", linewidth=1.4, label="max")
    ax.set_xlabel("Requests per query")
    ax.set_ylabel("Query completion latency (ms)")
    ax.set_title("Lance separate: 1KB request fanout sweep")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sweep 1KB separate-request fanout on the modified Lance scheduler path."
    )
    ap.add_argument(
        "--bin",
        default=str(_default_bin()),
        help="Path to lance_scheduler_qps_s3 from the modified Lance checkout.",
    )
    ap.add_argument(
        "--uri",
        default=os.environ.get("LANCE_QPS_URI"),
        help="Object URI to probe, e.g. s3://bucket/path/file.bin",
    )
    ap.add_argument(
        "--fanouts",
        default="1-128",
        help="Comma-separated ints and ranges, e.g. 1-16,32,64,128",
    )
    ap.add_argument(
        "--nlist",
        type=int,
        default=128,
        help="Synthetic cluster count. Keep this at least max(fanout).",
    )
    ap.add_argument(
        "--cluster-bytes",
        type=int,
        default=1024,
        help="Synthetic cluster size in bytes. Default: 1024.",
    )
    ap.add_argument(
        "--pool-size",
        type=int,
        default=128,
        help="LANCE_PROCESS_IO_THREADS_LIMIT. Default: 128.",
    )
    ap.add_argument(
        "--num-queries",
        type=int,
        default=40,
        help="Measured queries per fanout. Default: 40.",
    )
    ap.add_argument(
        "--warmup-queries",
        type=int,
        default=8,
        help="Warmup queries per fanout. Default: 8.",
    )
    ap.add_argument(
        "--query-concurrency",
        type=int,
        default=1,
        help="Benchmark-level query concurrency. Default: 1.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed. Default: 42.",
    )
    ap.add_argument(
        "--preplan",
        action="store_true",
        help="Preview request plans before submission.",
    )
    ap.add_argument(
        "--max-iop-size",
        type=int,
        default=1024 * 1024 * 1024,
        help="LANCE_MAX_IOP_SIZE. Default: 1 GiB.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: results/lance_1kb_fanout_<timestamp>",
    )
    ap.add_argument(
        "--query-ranges-dir",
        default=None,
        help="Optional directory containing per-fanout query JSONs named fanout_XXX_queries.json.",
    )
    ap.add_argument(
        "--record-per-iop",
        action="store_true",
        help="Also export a per-I/O CSV for each fanout run.",
    )
    args = ap.parse_args()
    args.fanouts = _parse_int_spec(args.fanouts)
    if not args.uri:
        ap.error("--uri is required (or set LANCE_QPS_URI)")
    if args.nlist < max(args.fanouts):
        ap.error("--nlist must be >= max fanout")
    if int(args.query_concurrency) != 1:
        ap.error("--query-concurrency must be 1 for this single-query fanout experiment")
    if args.query_ranges_dir:
        qdir = Path(args.query_ranges_dir).resolve()
        if not qdir.is_dir():
            ap.error(f"--query-ranges-dir not found: {qdir}")
        for fanout in args.fanouts:
            p = qdir / f"fanout_{fanout:03d}_queries.json"
            if not p.exists():
                ap.error(f"missing query file for fanout {fanout}: {p}")
    return args


def main() -> int:
    args = parse_args()
    bin_path = Path(args.bin).resolve()
    if not bin_path.exists():
        print(
            f"binary not found: {bin_path}\n"
            f"build it in the modified Lance checkout (default: {DEFAULT_LANCE_ROOT}) with:\n"
            f"  cargo build --release --example lance_scheduler_qps_s3",
            file=sys.stderr,
        )
        return 2

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (ROOT / "results" / f"lance_1kb_fanout_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunMetrics] = []
    for idx, fanout in enumerate(args.fanouts):
        print(
            f"[run] fanout={fanout} pool={args.pool_size} cluster_bytes={args.cluster_bytes} "
            f"queries={args.num_queries}",
            flush=True,
        )
        row = _run_one(
            bin_path=bin_path,
            uri=args.uri,
            out_dir=out_dir,
            fanout=fanout,
            nlist=args.nlist,
            cluster_bytes=args.cluster_bytes,
            pool_size=args.pool_size,
            num_queries=args.num_queries,
            warmup_queries=args.warmup_queries,
            query_concurrency=args.query_concurrency,
            seed=args.seed + idx,
            preplan=bool(args.preplan),
            max_iop_size=args.max_iop_size,
            query_ranges_file=(
                Path(args.query_ranges_dir).resolve() / f"fanout_{fanout:03d}_queries.json"
                if args.query_ranges_dir
                else None
            ),
            record_per_iop=bool(args.record_per_iop),
        )
        rows.append(row)
        print(
            f"[ok] fanout={row.fanout} avg={row.avg_latency_ms:.3f}ms "
            f"p95={row.p95_latency_ms:.3f}ms max={row.max_latency_ms:.3f}ms "
            f"planned={row.planned_ranges_avg:.2f}",
            flush=True,
        )

    csv_path = out_dir / "summary.csv"
    _write_csv(rows, csv_path)
    _write_readme(out_path=out_dir / "README.txt", args=args, rows=rows, bin_path=bin_path)
    _maybe_plot(rows, out_dir / "latency_vs_fanout.png")

    print(f"[done] wrote summary to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

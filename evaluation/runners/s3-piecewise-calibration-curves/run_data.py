#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE_DIR = ROOT / "experiments/figure-04-object-store-io/stages" / HERE.name
DATA_DIR = STAGE_DIR / "data" / "latest"
PAPER_DIR = STAGE_DIR / "data" / "paper"
GENERATED_DIR = DATA_DIR / "generated"

DEFAULT_CALIBRATION_JSON = ROOT / "results" / "pipeline" / "sift1m_sigmod_paper_20260222_023159" / "stage0" / "calibration.json"
DEFAULT_PROBE_DIR = ROOT / "results" / "model_probe" / "sigmod_20260222_023159"
DEFAULT_PROBE_OUT = GENERATED_DIR / "probe_run"
# Overhead calibration inputs. A plain invocation copies the frozen dense
# probe into data/latest/ for an offline reproduction. serial_stats.json is
# produced only by this runner's live-S3 probe.
DEFAULT_DENSE_PROBE_CSV = PAPER_DIR / "dense_probe.csv"
DEFAULT_SERVER_MODEL_JSON = PAPER_DIR / "source_latency_model.json"
DEFAULT_SERVER_SUMMARY_JSON = PAPER_DIR / "source_calibration_summary.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare Figure 3 data under runners.")
    ap.add_argument(
        "--probe-stats-json",
        default=None,
        help="Optional serial_stats.json to convert instead of copying the frozen paper dense probe.",
    )
    ap.add_argument("--latency-model-json", default=str(DEFAULT_SERVER_MODEL_JSON))
    ap.add_argument("--calibration-summary-json", default=str(DEFAULT_SERVER_SUMMARY_JSON))
    ap.add_argument("--rerun-probe", action="store_true", help="Run the experiment-owned S3 probe before freezing data.")
    ap.add_argument("--probe-out-dir", default=str(DEFAULT_PROBE_OUT))
    ap.add_argument(
        "--config",
        default=None,
        help="Optional YAML containing s3.bucket/region; CLI flags and VECIO_S3_* avoid a repo-local config.",
    )
    ap.add_argument("--s3-bucket", default=None, help="S3 bucket for a live probe (otherwise VECIO_S3_BUCKET).")
    ap.add_argument("--s3-region", default=None, help="AWS region for a live probe (otherwise VECIO_S3_REGION/AWS_DEFAULT_REGION).")
    ap.add_argument("--object-size-mb", type=int, default=256)
    ap.add_argument("--sizes-linear-min-mib", type=float, default=0.5)
    ap.add_argument("--sizes-linear-max-mib", type=float, default=16.0)
    ap.add_argument("--sizes-linear-step-mib", type=float, default=0.5)
    ap.add_argument("--serial-warmup", type=int, default=5)
    ap.add_argument("--serial-samples", type=int, default=20)
    ap.add_argument("--serial-rounds", type=int, default=1)
    ap.add_argument("--between-rounds-sleep-ms", type=int, default=200)
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check-only", action="store_true")
    return ap.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_empty_anchor_points(out_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "size_bytes", "p50_ms", "p95_ms"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _export_dense_probe(probe_stats_json: Path, out_csv: Path) -> list[dict[str, Any]]:
    serial_stats = _load_json(probe_stats_json)
    rows: list[dict[str, Any]] = []
    for rec in serial_stats:
        if not isinstance(rec, dict):
            continue
        rows.append(
            {
                "size_bytes": int(rec["size_bytes"]),
                "n": int(rec["n"]),
                "mean_ms": float(rec["mean_ms"]),
                "p50_ms": float(rec["p50_ms"]),
                "p95_ms": float(rec["p95_ms"]),
                "p99_ms": float(rec["p99_ms"]),
                "std_ms": float(rec["std_ms"]),
            }
        )
    rows.sort(key=lambda row: row["size_bytes"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["size_bytes", "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "std_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _copy_dense_probe(source_csv: Path, out_csv: Path) -> list[dict[str, Any]]:
    with source_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(source_csv.read_text(encoding="utf-8"), encoding="utf-8")
    return rows


def _build_probe_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(HERE / "probe_s3_range_cost_model.py"),
    ]
    if args.config:
        cmd.extend(["--config", str(_resolve(args.config))])
    if args.s3_bucket:
        cmd.extend(["--s3-bucket", str(args.s3_bucket)])
    if args.s3_region:
        cmd.extend(["--s3-region", str(args.s3_region)])
    cmd.extend(
        [
            "--object-size-mb",
            str(int(args.object_size_mb)),
            "--sizes-linear-min-mib",
            str(float(args.sizes_linear_min_mib)),
            "--sizes-linear-max-mib",
            str(float(args.sizes_linear_max_mib)),
            "--sizes-linear-step-mib",
            str(float(args.sizes_linear_step_mib)),
            "--serial-warmup",
            str(int(args.serial_warmup)),
            "--serial-samples",
            str(int(args.serial_samples)),
            "--serial-rounds",
            str(int(args.serial_rounds)),
            "--between-rounds-sleep-ms",
            str(int(args.between_rounds_sleep_ms)),
            "--skip-concurrency",
            "--max-workers",
            str(int(args.max_workers)),
            "--seed",
            str(int(args.seed)),
            "--out",
            str(out_dir),
        ]
    )
    return cmd


def main() -> int:
    args = parse_args()
    probe_stats_json = _resolve(args.probe_stats_json) if args.probe_stats_json else None
    dense_probe_csv = DEFAULT_DENSE_PROBE_CSV
    latency_model_json = _resolve(args.latency_model_json)
    calibration_summary_json = _resolve(args.calibration_summary_json)
    probe_out_dir = _resolve(args.probe_out_dir)

    planned_cmd = None
    if args.rerun_probe:
        planned_cmd = _build_probe_command(args, probe_out_dir)
        probe_stats_json = probe_out_dir / "serial_stats.json"

    if args.check_only:
        if probe_stats_json is not None:
            print(f"[figure3] probe_stats_json={probe_stats_json}")
        else:
            print(f"[figure3] frozen_dense_probe_csv={dense_probe_csv}")
        print(f"[figure3] latency_model_json={latency_model_json}")
        print(f"[figure3] calibration_summary_json={calibration_summary_json}")
        if planned_cmd is not None:
            print("[figure3] probe_command=")
            print(" ".join(planned_cmd))
        return 0

    if planned_cmd is not None:
        probe_out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(planned_cmd, check=True, cwd=str(ROOT))

    if probe_stats_json is not None and not probe_stats_json.exists():
        raise SystemExit(f"Missing probe serial_stats.json: {probe_stats_json}")
    if probe_stats_json is None and not dense_probe_csv.exists():
        raise SystemExit(f"Missing frozen dense probe CSV: {dense_probe_csv}")
    if not latency_model_json.exists():
        raise SystemExit(f"Missing latency model json: {latency_model_json}")
    if not calibration_summary_json.exists():
        raise SystemExit(f"Missing calibration summary json: {calibration_summary_json}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    anchor_rows = _write_empty_anchor_points(DATA_DIR / "anchor_points.csv")
    if probe_stats_json is not None:
        dense_rows = _export_dense_probe(probe_stats_json, DATA_DIR / "dense_probe.csv")
        dense_probe_source = str(probe_stats_json)
    else:
        dense_rows = _copy_dense_probe(dense_probe_csv, DATA_DIR / "dense_probe.csv")
        dense_probe_source = str(dense_probe_csv)
    (DATA_DIR / "source_latency_model.json").write_text(
        latency_model_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (DATA_DIR / "source_calibration_summary.json").write_text(
        calibration_summary_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    provenance = {
        "probe_stats_json": str(probe_stats_json) if probe_stats_json is not None else None,
        "dense_probe_source": dense_probe_source,
        "latency_model_json": str(latency_model_json),
        "calibration_summary_json": str(calibration_summary_json),
        "rerun_probe": bool(args.rerun_probe),
        "anchor_points": len(anchor_rows),
        "dense_probe_points": len(dense_rows),
        "probe_command": planned_cmd,
        "note": "Figure 3 now freezes the server-side single-request 1-16 MiB calibration with a 3 MiB breakpoint; legacy anchor-based fit data is intentionally excluded.",
    }
    (DATA_DIR / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"[ok] anchor_csv={DATA_DIR / 'anchor_points.csv'}")
    print(f"[ok] dense_probe_csv={DATA_DIR / 'dense_probe.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

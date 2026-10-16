#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARE = HERE / "prepare_datasets.py"
CLOSED_LOOP_RUNNER = ROOT / "runners" / "figure-10-p-sweep-closedloop" / "run_data.py"

DEFAULT_DATASETS = "sift1m,glove,deep1m,openimage,msong,random1m128"
DEFAULT_DATA_ROOT = ROOT / "experiments" / HERE.name / "data" / "latest"
DEFAULT_LANCE_ROOT = Path(os.environ.get("VECIO_LANCE_ROOT", str(ROOT.parent / "lance")))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Figure 7 style QPS-vs-savings on extra 1M-scale datasets.")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS)
    ap.add_argument("--nlist", type=int, default=0, help="If >0, forward to closed-loop (override preset default).")
    ap.add_argument("--nprobe", type=int, default=0, help="If >0, forward to closed-loop (override preset default).")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--remote-work-dir", default="/tmp/warp_qps_vs_savings_q4io64_more_1m")
    ap.add_argument("--local-lance-root", default=str(DEFAULT_LANCE_ROOT))
    ap.add_argument("--ssh-target", default="", help="If empty, the inner runner runs locally (default). Set to ubuntu@<host> to delegate.")
    ap.add_argument("--ssh-key", default="")
    ap.add_argument("--s3-bucket", default="", help="Forwarded to closed-loop runner.")
    ap.add_argument("--s3-region", default="", help="Forwarded to closed-loop runner.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-queries", type=int, default=120)
    ap.add_argument("--warmup-queries", type=int, default=20)
    ap.add_argument("--query-subset-mode", default="random", choices=["random", "first"])
    ap.add_argument("--query-subset-seed", type=int, default=12345)
    ap.add_argument("--train-size", type=int, default=0)
    ap.add_argument("--environments", default="q4-io64")
    ap.add_argument("--balanced-ps", default="16")
    ap.add_argument("--ks", default="8,16,32,64,96,128")
    ap.add_argument("--greedy-gap-kb", default="64,128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--greedy-span-caps-mb", default="1,2,4,8,16,32")
    ap.add_argument("--prepare-download", action="store_true")
    ap.add_argument("--prepare-force", action="store_true")
    ap.add_argument("--skip-unavailable", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-remote", action="store_true")
    ap.add_argument("--reuse-query-ranges", action="store_true")
    ap.add_argument("--reuse-remote-workspace", action="store_true")
    ap.add_argument("--skip-remote-build", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ready_datasets(stats_json: Path, requested: list[str], *, skip_unavailable: bool) -> list[str]:
    if not stats_json.exists():
        return requested
    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    ready = [str(r["dataset"]) for r in rows if r.get("status") == "ready"]
    missing = [d for d in requested if d not in ready]
    if missing and not skip_unavailable:
        raise SystemExit(f"datasets not ready: {','.join(missing)}")
    return [d for d in requested if d in ready]


def main() -> int:
    args = parse_args()
    requested = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    data_root = Path(args.data_root).expanduser().resolve()
    stats_json = data_root / "dataset_stats.json"
    stats_csv = data_root / "dataset_stats.csv"

    if args.check_only:
        print(f"[plan] datasets={requested}")
        print(f"[plan] data_root={data_root}")
        print(f"[plan] skip_prepare={bool(args.skip_prepare)}")
        print(f"[plan] skip_remote={bool(args.skip_remote)}")
        print(f"[plan] local_lance_root={Path(args.local_lance_root).expanduser().resolve()}")
        return 0

    if not args.skip_prepare:
        prep_cmd = [
            sys.executable,
            str(PREPARE),
            "--datasets",
            ",".join(requested),
            "--stats-json",
            str(stats_json),
            "--stats-csv",
            str(stats_csv),
        ]
        if args.prepare_download:
            prep_cmd.append("--download")
        if args.prepare_force:
            prep_cmd.append("--force")
        if args.skip_unavailable:
            prep_cmd.append("--skip-unavailable")
        _run(prep_cmd)

    datasets = _ready_datasets(stats_json, requested, skip_unavailable=bool(args.skip_unavailable))
    if not datasets:
        raise SystemExit("no prepared datasets available to run")

    cmd = [
        sys.executable,
        str(CLOSED_LOOP_RUNNER),
        "--datasets",
        ",".join(datasets),
        "--environments",
        str(args.environments),
        "--num-queries",
        str(int(args.num_queries)),
        "--warmup-queries",
        str(int(args.warmup_queries)),
        "--query-subset-mode",
        str(args.query_subset_mode),
        "--query-subset-seed",
        str(int(args.query_subset_seed)),
        "--train-size",
        str(int(args.train_size)),
        "--balanced-ps",
        str(args.balanced_ps),
        "--ks",
        str(args.ks),
        "--greedy-gap-kb",
        str(args.greedy_gap_kb),
        "--greedy-span-caps-mb",
        str(args.greedy_span_caps_mb),
        "--repeats",
        str(args.repeats),
        "--remote-work-dir",
        args.remote_work_dir,
        "--data-root",
        str(data_root),
        "--local-lance-root",
        str(Path(args.local_lance_root).expanduser().resolve()),
        "--ssh-target",
        args.ssh_target,
        "--ssh-key",
        args.ssh_key,
    ]
    if args.reuse_query_ranges:
        cmd.append("--reuse-query-ranges")
    if args.reuse_remote_workspace:
        cmd.append("--reuse-remote-workspace")
    if args.skip_remote_build:
        cmd.append("--skip-remote-build")
    if args.skip_remote:
        cmd.append("--skip-remote")
    if args.check_only:
        cmd.append("--check-only")
    if args.force:
        cmd.append("--force")
    if str(args.s3_bucket).strip():
        cmd.extend(["--s3-bucket", str(args.s3_bucket).strip()])
    if str(args.s3_region).strip():
        cmd.extend(["--s3-region", str(args.s3_region).strip()])
    if int(args.nlist) > 0:
        cmd.extend(["--nlist", str(int(args.nlist))])
    if int(args.nprobe) > 0:
        cmd.extend(["--nprobe", str(int(args.nprobe))])
    _run(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

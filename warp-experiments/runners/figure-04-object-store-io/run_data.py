#!/usr/bin/env python3
"""Collect Figure 4's four object-store runtime topologies in one command."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT / "runners/s3-piecewise-calibration-curves/probe_s3_range_cost_model.py"
OUT = ROOT / "experiments/figure-04-object-store-io/data/latest/runtime"
FANOUT_SOURCE = (
    ROOT
    / "experiments/figure-04-object-store-io/stages/real-s3-tiny-request-fanout-sanity"
    / "data/latest/summary.csv"
)
FANOUT_DESTINATION = (
    ROOT / "experiments/figure-04-object-store-io/data/latest/request_fanout.csv"
)
SIZES = tuple(int(value * 2**20) for value in (0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 16, 24, 32))
EXPECTED = {
    "ap_same_region": "ap_same_region_serial_stats.json",
    "us_same_region": "us_same_region_serial_stats.json",
    "cross_region": "cross_region_serial_stats.json",
    "public_internet": "public_internet_serial_stats.json",
}


@dataclass(frozen=True)
class Topology:
    name: str
    target: str
    bucket: str
    region: str
    samples: int


def _env_topologies() -> list[str]:
    values = []
    for name, env_name in (
        ("ap_same_region", "VECIO_FIG4_AP_SAME_REGION"),
        ("us_same_region", "VECIO_FIG4_US_SAME_REGION"),
        ("cross_region", "VECIO_FIG4_CROSS_REGION"),
        ("public_internet", "VECIO_FIG4_PUBLIC_INTERNET"),
    ):
        value = str(os.environ.get(env_name, "")).strip()
        if value:
            values.append(f"{name},{value}")
    return values


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--topology",
        action="append",
        default=[],
        metavar="NAME,TARGET,BUCKET,REGION,SAMPLES",
        help="TARGET is local or an SSH target. Repeat exactly four times.",
    )
    ap.add_argument("--remote-repo-root", default="~/warp-experiments")
    ap.add_argument("--ssh-key", default="")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--serial-warmup", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    return ap.parse_args()


def _parse_topologies(raw_values: list[str]) -> list[Topology]:
    output = []
    for raw in raw_values:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 5:
            raise SystemExit(f"invalid --topology {raw!r}; expected five comma-separated fields")
        name, target, bucket, region, samples = parts
        if name not in EXPECTED:
            raise SystemExit(f"unknown topology {name!r}; expected {sorted(EXPECTED)}")
        if not target or not bucket or not region:
            raise SystemExit(f"invalid --topology {raw!r}; TARGET, BUCKET, and REGION are required")
        sample_count = int(samples)
        if sample_count <= 0:
            raise SystemExit(f"invalid --topology {raw!r}; SAMPLES must be positive")
        output.append(Topology(name, target, bucket, region, sample_count))
    names = [item.name for item in output]
    if set(names) != set(EXPECTED) or len(names) != len(set(names)):
        missing = sorted(set(EXPECTED) - set(names))
        duplicate = sorted(name for name in set(names) if names.count(name) > 1)
        raise SystemExit(f"Figure 4 requires exactly four topologies; missing={missing}, duplicate={duplicate}")
    return output


def _probe_args(topology: Topology, out_dir: str, serial_warmup: int) -> list[str]:
    return [
        sys.executable,
        str(PROBE),
        "--s3-bucket", topology.bucket,
        "--s3-region", topology.region,
        "--object-size-mb", "64",
        "--sizes-bytes", ",".join(map(str, SIZES)),
        "--serial-warmup", str(serial_warmup),
        "--serial-samples", str(topology.samples),
        "--serial-rounds", "1",
        "--skip-concurrency",
        "--seed", "7",
        "--out", out_dir,
    ]


def _validate(path: Path, samples: int) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if [int(row["size_bytes"]) for row in rows] != list(SIZES):
        raise ValueError(f"{path}: request-size ladder does not match the paper")
    if any(int(row["n"]) != samples for row in rows):
        raise ValueError(f"{path}: sample count does not match requested n={samples}")
    required = {"size_bytes", "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "std_ms"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"{path}: incomplete serial-stat schema")


def _local(topology: Topology, work: Path, serial_warmup: int) -> Path:
    command = _probe_args(topology, str(work), serial_warmup)
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return work / "serial_stats.json"


def _remote(topology: Topology, work: Path, args: argparse.Namespace) -> Path:
    remote_out = f"/tmp/warp_fig4_{topology.name}"
    probe_args = _probe_args(topology, remote_out, int(args.serial_warmup))
    remote_probe = f"{args.remote_repo_root.rstrip('/')}/runners/s3-piecewise-calibration-curves/probe_s3_range_cost_model.py"
    probe_args[1] = remote_probe
    remote_command = f"cd {shlex.quote(args.remote_repo_root)} && {shlex.join(probe_args)}"
    ssh = ["ssh", "-o", "BatchMode=yes"]
    scp = ["scp", "-q"]
    if args.ssh_key:
        ssh.extend(["-i", args.ssh_key])
        scp.extend(["-i", args.ssh_key])
    print("$ " + shlex.join([*ssh, topology.target, remote_command]), flush=True)
    subprocess.run([*ssh, topology.target, remote_command], check=True)
    work.mkdir(parents=True, exist_ok=True)
    local = work / "serial_stats.json"
    print("$ " + shlex.join([*scp, f"{topology.target}:{remote_out}/serial_stats.json", str(local)]), flush=True)
    subprocess.run([*scp, f"{topology.target}:{remote_out}/serial_stats.json", str(local)], check=True)
    return local


def main() -> int:
    args = parse_args()
    raw = list(args.topology) or _env_topologies()
    if args.check_only and not raw:
        print("[error] missing Figure 4 topology configuration", file=sys.stderr)
        for name in EXPECTED:
            print(f"[required] {name},TARGET,BUCKET,REGION,SAMPLES", file=sys.stderr)
        raise SystemExit(2)
    topologies = _parse_topologies(raw)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.check_only:
        for item in topologies:
            print(
                f"[plan] {item.name}: target={item.target} bucket={item.bucket} "
                f"region={item.region} samples={item.samples}"
            )
        print(f"[plan] out_dir={out_dir}")
        print(f"[plan] request_fanout={FANOUT_SOURCE} -> {FANOUT_DESTINATION}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    for topology in topologies:
        destination = out_dir / EXPECTED[topology.name]
        if destination.is_file() and not args.force:
            _validate(destination, topology.samples)
            print(f"[skip] valid topology {topology.name}")
        else:
            work = out_dir / "work" / topology.name
            source = (
                _local(topology, work, int(args.serial_warmup))
                if topology.target == "local"
                else _remote(topology, work, args)
            )
            _validate(source, topology.samples)
            shutil.copy2(source, destination)
        provenance.append(
            {
                "name": topology.name,
                "target": topology.target,
                "bucket": topology.bucket,
                "region": topology.region,
                "samples": topology.samples,
                "output": destination.name,
            }
        )
    (out_dir.parent / "topology_provenance.json").write_text(
        json.dumps({"topologies": provenance}, indent=2) + "\n", encoding="utf-8"
    )
    if not FANOUT_SOURCE.is_file():
        raise SystemExit(
            "Figure 4 fanout stage output is missing: "
            f"{FANOUT_SOURCE}. Run the registered dependency first."
        )
    FANOUT_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FANOUT_SOURCE, FANOUT_DESTINATION)
    print(f"[ok] Figure 4 four-topology evidence: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

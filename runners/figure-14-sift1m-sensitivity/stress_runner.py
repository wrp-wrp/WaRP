#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA_ROOT = (
    ROOT
    / "experiments"
    / "figure-14-sift1m-sensitivity"
    / "data"
    / "latest"
    / "stress"
)
LANCE_ROOT = ROOT / "third_party" / "lance"
EXPORT_SCRIPT = ROOT / "runners" / "shared" / "export_real_query_ranges_for_lance.py"
FIG3_FIT_SUMMARY = (
    ROOT
    / "experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves"
    / "data/paper/fit_summary.json"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the SIFT1M sensitivity real-S3 stress stage.")
    ap.add_argument("--datasets", default="sift1m,glove")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=256)
    ap.add_argument("--num-queries", type=int, default=120)
    ap.add_argument("--warmup-queries", type=int, default=20)
    ap.add_argument("--query-subset-mode", default="random", choices=["random", "first"])
    ap.add_argument("--query-subset-seed", type=int, default=12345)
    ap.add_argument("--query-concurrency", type=int, default=8)
    ap.add_argument("--s3-bucket", default="your-s3-bucket-name")
    ap.add_argument("--s3-region", default="ap-northeast-1")
    ap.add_argument("--ks", default="8,16,32,64,96,128")
    ap.add_argument("--greedy-gap-kb", default="64,128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--greedy-span-caps-mb", default="1,2,4,8,16,32")
    ap.add_argument("--balanced-ps", default="1")
    ap.add_argument("--balanced-exact-requests", action="store_true")
    ap.add_argument("--process-io-threads-limit", type=int, default=128)
    ap.add_argument("--cloud-io-parallelism", type=int, default=128)
    ap.add_argument("--max-iop-size-bytes", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--train-size", type=int, default=200000)
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="Directory for query_ranges/, remote_runs/, and provenance outputs.")
    ap.add_argument("--ssh-target", default="")
    ap.add_argument("--ssh-key", default="")
    ap.add_argument("--remote-work-dir", default="/tmp/warp_sift1m_real_s3_sensitivity")
    ap.add_argument("--local-lance-root", default=str(LANCE_ROOT))
    ap.add_argument("--local-vecio-root", default=str(ROOT))
    ap.add_argument("--remote-export", action="store_true")
    ap.add_argument("--reuse-remote-workspace", action="store_true")
    ap.add_argument("--skip-remote-build", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-remote", action="store_true")
    ap.add_argument("--latency-model-json-in", default="")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _parse_csv_list(raw: str) -> List[str]:
    out = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    if not out:
        raise ValueError("empty csv list")
    return out


def _parse_int_csv(raw: str) -> List[int]:
    return sorted({int(tok.strip()) for tok in str(raw).split(",") if tok.strip()})


def _parse_float_csv(raw: str) -> List[float]:
    values = sorted({float(tok.strip()) for tok in str(raw).split(",") if tok.strip()})
    if not values:
        raise ValueError("empty float csv list")
    return values


def _run(cmd: List[str], *, cwd: Path | None = None, check_only: bool = False, env: dict | None = None) -> None:
    print("$ " + " ".join(cmd))
    if check_only:
        return
    retries = 4 if cmd and cmd[0] in {"ssh", "rsync"} else 1
    delay_s = 3.0
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            print(f"[retry] attempt {attempt}/{retries} failed; sleeping {delay_s:.0f}s", flush=True)
            time.sleep(delay_s)
            delay_s *= 2.0
    if last_error is not None:
        raise last_error


def _run_capture(cmd: List[str], *, cwd: Path | None = None, check_only: bool = False) -> str:
    print("$ " + " ".join(cmd))
    if check_only:
        return ""
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True)


def _remote_path_exists(args: argparse.Namespace, remote_path: str) -> bool:
    out = _run_capture(
        _remote_cmd(args, f"test -e {remote_path} && echo yes || echo no"),
        check_only=bool(args.check_only),
    ).strip()
    return out == "yes"


def _ensure_clean_dir(path: Path, *, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _query_dir() -> Path:
    return DATA_ROOT / "query_ranges"


def _remote_dir() -> Path:
    return DATA_ROOT / "remote_runs"


def _aggregated_json() -> Path:
    return DATA_ROOT / "aggregated_results.json"


def _aggregated_csv() -> Path:
    return DATA_ROOT / "aggregated_results.csv"


def _latency_model_json() -> Path:
    return DATA_ROOT / "latency_model.json"


def _provenance_json() -> Path:
    return DATA_ROOT / "data_provenance.json"


def _write_latency_model_json(out_path: Path) -> dict:
    fit = json.loads(FIG3_FIT_SUMMARY.read_text(encoding="utf-8"))
    piecewise = fit["piecewise_continuous"]
    payload = {
        "latency_model": "piecewise_linear",
        "latency_pw_breaks_bytes": [int(round(float(piecewise["break_bytes"])))],
        "latency_pw_overheads_ms": [float(v) for v in piecewise["overheads_ms"]],
        "latency_pw_rates_ms_per_byte": [float(v) for v in piecewise["rates_ms_per_byte"]],
        "lat_req_ms": float(piecewise["overheads_ms"][0]),
        "lat_byte_ms": float(piecewise["rates_ms_per_byte"][0]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _resolve_latency_model(args: argparse.Namespace, out_path: Path) -> dict:
    custom = str(args.latency_model_json_in).strip()
    if not custom:
        return _write_latency_model_json(out_path)
    src = Path(custom).expanduser().resolve()
    payload = json.loads(src.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _export_query_ranges(args: argparse.Namespace, dataset: str, reorder_method: str, out_path: Path) -> None:
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--dataset",
        dataset,
        "--nlist",
        str(int(args.nlist)),
        "--nprobe",
        str(int(args.nprobe)),
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
        "--reorder-method",
        reorder_method,
        "--s3-bucket",
        str(args.s3_bucket),
        "--s3-region",
        str(args.s3_region),
        "--out",
        str(out_path),
    ]
    _run(cmd, cwd=ROOT, check_only=bool(args.check_only))


def _ssh_prefix(ssh_target: str, ssh_key: str) -> List[str]:
    cmd = ["ssh"]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    cmd.extend(["-o", "StrictHostKeyChecking=accept-new", ssh_target])
    return cmd


def _rsync_ssh_value(ssh_key: str) -> str:
    parts = ["ssh"]
    if ssh_key:
        parts.extend(["-i", ssh_key])
    parts.extend(["-o", "StrictHostKeyChecking=accept-new"])
    return " ".join(parts)


def _rsync_to_remote(local_path: str | Path, remote_spec: str, ssh_key: str, *, delete: bool, excludes: List[str] | None, check_only: bool) -> None:
    cmd = ["rsync", "-az"]
    if delete:
        cmd.append("--delete")
    for pattern in excludes or []:
        cmd.extend(["--exclude", pattern])
    cmd.extend(["-e", _rsync_ssh_value(ssh_key), str(local_path), remote_spec])
    _run(cmd, check_only=check_only)


def _rsync_from_remote(remote_spec: str, local_path: Path, ssh_key: str, *, delete: bool, check_only: bool) -> None:
    cmd = ["rsync", "-az"]
    if delete:
        cmd.append("--delete")
    cmd.extend(["-e", _rsync_ssh_value(ssh_key), remote_spec, str(local_path)])
    _run(cmd, check_only=check_only)


def _remote_cmd(args: argparse.Namespace, script: str) -> List[str]:
    return _ssh_prefix(str(args.ssh_target), str(args.ssh_key)) + [script]


def _remote_export_query_ranges(
    args: argparse.Namespace,
    datasets: List[str],
    remote_root: str,
    *,
    query_dir: Path,
) -> None:
    local_vecio_root = Path(str(args.local_vecio_root)).resolve()
    remote_vecio = f"{remote_root}/vecio"
    remote_inputs = f"{remote_root}/inputs"

    if not bool(args.reuse_remote_workspace):
        _run(
            _remote_cmd(args, f"mkdir -p {remote_root} {remote_inputs}/query_ranges && rm -rf {remote_vecio} && mkdir -p {remote_vecio}/data/processed"),
            check_only=bool(args.check_only),
        )
    else:
        _run(
            _remote_cmd(args, f"mkdir -p {remote_root} {remote_inputs}/query_ranges {remote_vecio}/data/processed"),
            check_only=bool(args.check_only),
        )

    _rsync_to_remote(
        local_vecio_root.as_posix() + "/",
        f"{args.ssh_target}:{remote_vecio}/",
        str(args.ssh_key),
        delete=True,
        excludes=[".git", ".venv", "__pycache__", "results", "paper_experiments", "data"],
        check_only=bool(args.check_only),
    )
    for dataset in datasets:
        processed_src = local_vecio_root / "data" / "processed" / dataset
        if not processed_src.exists():
            raise SystemExit(f"missing processed dataset for remote export: {processed_src}")
        _rsync_to_remote(
            processed_src.as_posix() + "/",
            f"{args.ssh_target}:{remote_vecio}/data/processed/{dataset}/",
            str(args.ssh_key),
            delete=True,
            excludes=None,
            check_only=bool(args.check_only),
        )
        for reorder_method, suffix in [("identity", "original"), ("hilbert", "reordered")]:
            out_json = f"{remote_inputs}/query_ranges/{dataset}_{suffix}.json"
            if bool(args.reuse_remote_workspace):
                exists = _run_capture(
                    _remote_cmd(args, f"test -f {out_json} && echo yes || echo no"),
                    check_only=bool(args.check_only),
                ).strip()
                if exists == "yes":
                    continue
            cmd = _remote_cmd(
                args,
                " ".join(
                    [
                        f"PY={remote_vecio}/.venv/bin/python;",
                        'if [ ! -x "$PY" ]; then PY=python3; fi;',
                        f"export PYTHONPATH={remote_vecio}:$PYTHONPATH;",
                        f"cd {remote_vecio}",
                        "&&",
                        '"$PY"',
                        "runners/shared/export_real_query_ranges_for_lance.py",
                        f"--dataset {dataset}",
                        f"--nlist {int(args.nlist)}",
                        f"--nprobe {int(args.nprobe)}",
                        f"--num-queries {int(args.num_queries)}",
                        f"--warmup-queries {int(args.warmup_queries)}",
                        f"--query-subset-mode {args.query_subset_mode}",
                        f"--query-subset-seed {int(args.query_subset_seed)}",
                        f"--train-size {int(args.train_size)}",
                        f"--reorder-method {reorder_method}",
                        f"--s3-bucket {args.s3_bucket}",
                        f"--s3-region {args.s3_region}",
                        f"--out {out_json}",
                    ]
                ),
            )
            _run(cmd, check_only=bool(args.check_only))
    _rsync_from_remote(
        f"{args.ssh_target}:{remote_inputs}/query_ranges/",
        query_dir,
        str(args.ssh_key),
        delete=True,
        check_only=bool(args.check_only),
    )


def _prepare_remote_workspace(args: argparse.Namespace, local_lance_root: Path, *, datasets: List[str], query_dir: Path, latency_model_json: Path) -> Path:
    remote_root = str(args.remote_work_dir).rstrip("/")
    remote_lance = f"{remote_root}/lance"
    remote_inputs = f"{remote_root}/inputs"
    if not bool(args.reuse_remote_workspace):
        _run(
            _remote_cmd(args, f"mkdir -p {remote_root} {remote_inputs} && rm -rf {remote_lance}"),
            check_only=bool(args.check_only),
        )
    else:
        _run(_remote_cmd(args, f"mkdir -p {remote_root} {remote_inputs}"), check_only=bool(args.check_only))
    _rsync_to_remote(
        Path(str(local_lance_root)).resolve().as_posix() + "/",
        f"{args.ssh_target}:{remote_lance}/",
        str(args.ssh_key),
        delete=True,
        excludes=["target", ".git"],
        check_only=bool(args.check_only),
    )
    if bool(args.remote_export):
        _remote_export_query_ranges(args, datasets, remote_root, query_dir=query_dir)
    else:
        _rsync_to_remote(
            query_dir.resolve().as_posix() + "/",
            f"{args.ssh_target}:{remote_inputs}/query_ranges/",
            str(args.ssh_key),
            delete=True,
            excludes=None,
            check_only=bool(args.check_only),
        )
    _rsync_to_remote(
        latency_model_json.resolve(),
        f"{args.ssh_target}:{remote_inputs}/latency_model.json",
        str(args.ssh_key),
        delete=False,
        excludes=None,
        check_only=bool(args.check_only),
    )
    remote_bin = f"{remote_lance}/target/release/examples/strict_budget_real_s3"
    need_remote_build = not bool(args.skip_remote_build) or not _remote_path_exists(args, remote_bin)
    if need_remote_build:
        _run(
            _remote_cmd(
                args,
                " ".join(
                    [
                        f"cd {remote_lance}",
                        "&&",
                        "source $HOME/.cargo/env >/dev/null 2>&1 || true",
                        "&&",
                        "cargo build --release --example strict_budget_real_s3 -p lance-io",
                    ]
                ),
            ),
            check_only=bool(args.check_only),
        )
    return Path(remote_root)


def _prepare_local_binary(args: argparse.Namespace, local_lance_root: Path) -> Path:
    local_bin = local_lance_root / "target" / "release" / "examples" / "strict_budget_real_s3"
    need_build = not local_bin.exists()
    if local_bin.exists():
        try:
            subprocess.run(
                [str(local_bin), "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            need_build = True
    if need_build:
        _run(
            [
                "bash",
                "-lc",
                "source $HOME/.cargo/env >/dev/null 2>&1 || true && cargo build --release --example strict_budget_real_s3 -p lance-io",
            ],
            cwd=local_lance_root,
            check_only=bool(args.check_only),
        )
    return local_bin


def _run_remote_repeats(args: argparse.Namespace, datasets: List[str], remote_root: Path, *, remote_dir: Path) -> None:
    remote_bin = f"{remote_root.as_posix()}/lance/target/release/examples/strict_budget_real_s3"
    remote_inputs = f"{remote_root.as_posix()}/inputs"
    remote_results = f"{remote_root.as_posix()}/results"
    _run(_remote_cmd(args, f"rm -rf {remote_results} && mkdir -p {remote_results}"), check_only=bool(args.check_only))
    for dataset in datasets:
        for repeat_idx in range(1, int(args.repeats) + 1):
            rep = f"repeat_{repeat_idx:02d}"
            out_json = f"{remote_results}/{dataset}_{rep}.json"
            out_csv = f"{remote_results}/{dataset}_{rep}.csv"
            cmd = _remote_cmd(
                args,
                " ".join(
                    [
                        f"AWS_DEFAULT_REGION={args.s3_region}",
                        remote_bin,
                        f"--dataset {dataset}",
                        f"--original-query-json {remote_inputs}/query_ranges/{dataset}_original.json",
                        f"--reordered-query-json {remote_inputs}/query_ranges/{dataset}_reordered.json",
                        f"--latency-model-json {remote_inputs}/latency_model.json",
                        f"--out-json {out_json}",
                        f"--out-csv {out_csv}",
                        f"--ks {args.ks}",
                        f"--greedy-gap-kb {args.greedy_gap_kb}",
                        f"--greedy-span-caps-mb {args.greedy_span_caps_mb}",
                        f"--num-queries {int(args.num_queries)}",
                        f"--warmup-queries {int(args.warmup_queries)}",
                        f"--query-concurrency {int(args.query_concurrency)}",
                        f"--balanced-ps {args.balanced_ps}",
                        "--balanced-exact-requests" if bool(args.balanced_exact_requests) else "",
                        f"--process-io-threads-limit {int(args.process_io_threads_limit)}",
                        f"--cloud-io-parallelism {int(args.cloud_io_parallelism)}",
                        f"--max-iop-size-bytes {int(args.max_iop_size_bytes)}",
                    ]
                ),
            )
            _run(cmd, check_only=bool(args.check_only))
    _rsync_from_remote(
        f"{args.ssh_target}:{remote_results}/",
        remote_dir,
        str(args.ssh_key),
        delete=True,
        check_only=bool(args.check_only),
    )


def _run_local_repeats(args: argparse.Namespace, datasets: List[str], local_lance_root: Path, *, query_dir: Path, latency_model_json: Path, remote_dir: Path) -> None:
    local_bin = _prepare_local_binary(args, local_lance_root)
    _ensure_clean_dir(remote_dir, force=True)
    env = dict(os.environ)
    env["AWS_DEFAULT_REGION"] = str(args.s3_region)
    for dataset in datasets:
        for repeat_idx in range(1, int(args.repeats) + 1):
            rep = f"repeat_{repeat_idx:02d}"
            out_json = remote_dir / f"{dataset}_{rep}.json"
            out_csv = remote_dir / f"{dataset}_{rep}.csv"
            cmd = [
                str(local_bin),
                "--dataset",
                dataset,
                "--original-query-json",
                str(query_dir / f"{dataset}_original.json"),
                "--reordered-query-json",
                str(query_dir / f"{dataset}_reordered.json"),
                "--latency-model-json",
                str(latency_model_json),
                "--out-json",
                str(out_json),
                "--out-csv",
                str(out_csv),
                "--ks",
                str(args.ks),
                "--greedy-gap-kb",
                str(args.greedy_gap_kb),
                "--greedy-span-caps-mb",
                str(args.greedy_span_caps_mb),
                "--num-queries",
                str(int(args.num_queries)),
                "--warmup-queries",
                str(int(args.warmup_queries)),
                "--query-concurrency",
                str(int(args.query_concurrency)),
                "--balanced-ps",
                str(args.balanced_ps),
                "--process-io-threads-limit",
                str(int(args.process_io_threads_limit)),
                "--cloud-io-parallelism",
                str(int(args.cloud_io_parallelism)),
                "--max-iop-size-bytes",
                str(int(args.max_iop_size_bytes)),
            ]
            if bool(args.balanced_exact_requests):
                cmd.append("--balanced-exact-requests")
            _run(cmd, check_only=bool(args.check_only), env=env)


def _median(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(statistics.median(xs))


def _canonical_method_name(raw: object) -> str:
    method = str(raw)
    if method == "native_unbounded":
        return "lance_native"
    return method


def _choose_best_greedy(rows: List[dict]) -> dict | None:
    feasible = [
        r
        for r in rows
        if float(r.get("coverage", 0.0)) >= 1.0 - 1e-12 and r.get("qps") is not None
    ]
    if not feasible:
        return None
    feasible.sort(
        key=lambda r: (
            -float(r["qps"]),
            float(r["avg_latency_ms"]),
            float(r["avg_num_requests"]),
        )
    )
    return feasible[0]


def _aggregate_repeat_docs(datasets: List[str], *, remote_dir: Path, aggregated_json: Path, aggregated_csv: Path) -> dict:
    docs = []
    for dataset in datasets:
        per_dataset = []
        for repeat_idx in range(1, len(list(remote_dir.glob(f"{dataset}_repeat_*.json"))) + 1):
            path = remote_dir / f"{dataset}_repeat_{repeat_idx:02d}.json"
            if path.exists():
                per_dataset.append(json.loads(path.read_text(encoding="utf-8")))
        if not per_dataset:
            raise SystemExit(f"no remote repeat json for dataset={dataset} under {remote_dir}")
        docs.append((dataset, per_dataset))

    out = {
        "experiment": "figure-14-sift1m-sensitivity",
        "nprobe": 256,
        "query_concurrency": int(docs[0][1][0].get("query_concurrency", 8)),
        "repeats": int(min(len(items) for _, items in docs)),
        "rows": [],
        "greedy_candidates": [],
    }

    for dataset, repeat_docs in docs:
        summary_groups: Dict[Tuple[str, str, int | None], List[dict]] = defaultdict(list)
        greedy_groups: Dict[Tuple[str, int, int, int], List[dict]] = defaultdict(list)
        for doc in repeat_docs:
            for row in doc.get("rows", []):
                summary_groups[(str(row.get("layout")), _canonical_method_name(row.get("method")), row.get("budget_k"))].append(row)
            for row in doc.get("greedy_candidates", []):
                greedy_groups[(str(row.get("layout")), int(row.get("budget_k")), int(row.get("gap_kb")), int(row.get("span_cap_bytes")))].append(row)

        aggregated_greedy: List[dict] = []
        selected_by_k: Dict[int, dict] = {}
        for (layout, budget_k, gap_kb, span_cap_bytes), rows in sorted(greedy_groups.items()):
            agg = {
                "dataset": dataset,
                "layout": layout,
                "budget_k": int(budget_k),
                "gap_kb": int(gap_kb),
                "span_cap_bytes": int(span_cap_bytes),
                "coverage": _median(r.get("coverage") for r in rows) or 0.0,
                "wall_ms": _median(r.get("wall_ms") for r in rows),
                "qps": _median(r.get("qps") for r in rows),
                "avg_latency_ms": _median(r.get("avg_latency_ms") for r in rows),
                "p95_latency_ms": _median(r.get("p95_latency_ms") for r in rows),
                "avg_num_requests": _median(r.get("avg_num_requests") for r in rows),
                "avg_planned_bytes": _median(r.get("avg_planned_bytes") for r in rows),
                "avg_model_work_ms": _median(r.get("avg_model_work_ms") for r in rows),
                "avg_model_bottleneck_ms": _median(r.get("avg_model_bottleneck_ms") for r in rows),
                "avg_model_objective_ms": _median(r.get("avg_model_objective_ms") for r in rows),
                "selected_for_curve": False,
            }
            aggregated_greedy.append(agg)
        for budget_k in sorted({int(r["budget_k"]) for r in aggregated_greedy}):
            best = _choose_best_greedy([r for r in aggregated_greedy if int(r["budget_k"]) == budget_k])
            if best is not None:
                best["selected_for_curve"] = True
                selected_by_k[int(budget_k)] = best
        out["greedy_candidates"].extend(aggregated_greedy)

        for (layout, method, budget_k), rows in sorted(summary_groups.items()):
            if method == "greedy_frontier":
                continue
            out["rows"].append(
                {
                    "dataset": dataset,
                    "layout": layout,
                    "method": method,
                    "budget_k": budget_k,
                    "status": str(rows[0].get("status", "")),
                    "coverage": _median(r.get("coverage") for r in rows) or 0.0,
                    "wall_ms": _median(r.get("wall_ms") for r in rows),
                    "qps": _median(r.get("qps") for r in rows),
                    "avg_latency_ms": _median(r.get("avg_latency_ms") for r in rows),
                    "p95_latency_ms": _median(r.get("p95_latency_ms") for r in rows),
                    "avg_num_requests": _median(r.get("avg_num_requests") for r in rows),
                    "avg_planned_bytes": _median(r.get("avg_planned_bytes") for r in rows),
                    "avg_model_work_ms": _median(r.get("avg_model_work_ms") for r in rows),
                    "avg_model_bottleneck_ms": _median(r.get("avg_model_bottleneck_ms") for r in rows),
                    "avg_model_objective_ms": _median(r.get("avg_model_objective_ms") for r in rows),
                    "best_gap_kb": None,
                    "best_span_cap_bytes": None,
                }
            )
        for budget_k, row in sorted(selected_by_k.items()):
            out["rows"].append(
                {
                    "dataset": dataset,
                    "layout": row["layout"],
                    "method": "greedy_frontier",
                    "budget_k": int(budget_k),
                    "status": "ok" if float(row["coverage"]) >= 1.0 - 1e-12 else "partial",
                    "coverage": float(row["coverage"]),
                    "wall_ms": row["wall_ms"],
                    "qps": row["qps"],
                    "avg_latency_ms": row["avg_latency_ms"],
                    "p95_latency_ms": row["p95_latency_ms"],
                    "avg_num_requests": row["avg_num_requests"],
                    "avg_planned_bytes": row["avg_planned_bytes"],
                    "avg_model_work_ms": row["avg_model_work_ms"],
                    "avg_model_bottleneck_ms": row["avg_model_bottleneck_ms"],
                    "avg_model_objective_ms": row["avg_model_objective_ms"],
                    "best_gap_kb": int(row["gap_kb"]),
                    "best_span_cap_bytes": int(row["span_cap_bytes"]),
                }
            )

    aggregated_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    lines = [
        "section,dataset,layout,method,budget_k,status,coverage,wall_ms,qps,avg_latency_ms,p95_latency_ms,avg_num_requests,avg_planned_bytes,avg_model_work_ms,avg_model_bottleneck_ms,avg_model_objective_ms,best_gap_kb,best_span_cap_bytes"
    ]
    for row in out["rows"]:
        lines.append(
            "summary,{dataset},{layout},{method},{budget_k},{status},{coverage:.6f},{wall_ms},{qps},{avg_latency_ms},{p95_latency_ms},{avg_num_requests},{avg_planned_bytes},{avg_model_work_ms},{avg_model_bottleneck_ms},{avg_model_objective_ms},{best_gap_kb},{best_span_cap_bytes}".format(
                dataset=row["dataset"],
                layout=row["layout"],
                method=row["method"],
                budget_k="" if row["budget_k"] is None else row["budget_k"],
                status=row["status"],
                coverage=float(row["coverage"]),
                wall_ms="" if row["wall_ms"] is None else f"{float(row['wall_ms']):.6f}",
                qps="" if row["qps"] is None else f"{float(row['qps']):.6f}",
                avg_latency_ms="" if row["avg_latency_ms"] is None else f"{float(row['avg_latency_ms']):.6f}",
                p95_latency_ms="" if row["p95_latency_ms"] is None else f"{float(row['p95_latency_ms']):.6f}",
                avg_num_requests="" if row["avg_num_requests"] is None else f"{float(row['avg_num_requests']):.6f}",
                avg_planned_bytes="" if row["avg_planned_bytes"] is None else f"{float(row['avg_planned_bytes']):.6f}",
                avg_model_work_ms="" if row.get("avg_model_work_ms") is None else f"{float(row['avg_model_work_ms']):.6f}",
                avg_model_bottleneck_ms="" if row.get("avg_model_bottleneck_ms") is None else f"{float(row['avg_model_bottleneck_ms']):.6f}",
                avg_model_objective_ms="" if row["avg_model_objective_ms"] is None else f"{float(row['avg_model_objective_ms']):.6f}",
                best_gap_kb="" if row.get("best_gap_kb") is None else int(row["best_gap_kb"]),
                best_span_cap_bytes="" if row["best_span_cap_bytes"] is None else int(row["best_span_cap_bytes"]),
            )
        )
    aggregated_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_provenance(args: argparse.Namespace, latency_model: dict, provenance_json: Path, latency_model_json: Path) -> None:
    payload = {
        "experiment": "figure-14-sift1m-sensitivity",
        "datasets": _parse_csv_list(args.datasets),
        "nlist": int(args.nlist),
        "nprobe": int(args.nprobe),
        "num_queries": int(args.num_queries),
        "warmup_queries": int(args.warmup_queries),
        "query_concurrency": int(args.query_concurrency),
        "query_subset_mode": str(args.query_subset_mode),
        "query_subset_seed": int(args.query_subset_seed),
        "s3_bucket": str(args.s3_bucket),
        "s3_region": str(args.s3_region),
        "ks": _parse_int_csv(args.ks),
        "greedy_gap_kb": _parse_int_csv(args.greedy_gap_kb),
        "greedy_span_caps_mb": _parse_int_csv(args.greedy_span_caps_mb),
        "balanced_ps": _parse_float_csv(args.balanced_ps),
        "balanced_exact_requests": bool(args.balanced_exact_requests),
        "process_io_threads_limit": int(args.process_io_threads_limit),
        "cloud_io_parallelism": int(args.cloud_io_parallelism),
        "max_iop_size_bytes": int(args.max_iop_size_bytes),
        "repeats": int(args.repeats),
        "train_size": int(args.train_size),
        "remote_export": bool(args.remote_export),
        "ssh_target": str(args.ssh_target),
        "latency_model_json": (
            str(latency_model_json.relative_to(HERE))
            if latency_model_json.is_relative_to(HERE)
            else str(latency_model_json)
        ),
        "latency_model": latency_model,
    }
    provenance_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = [d.lower() for d in _parse_csv_list(args.datasets)]
    # Default to local execution on the current host. Pass --ssh-target to
    # delegate the Lance benchmark to a separate EC2 box (production mode).
    if not args.ssh_target and not args.skip_remote:
        args.skip_remote = True

    # Redirect DATA_ROOT-derived paths when the parent sensitivity runner
    # isolates this benchmark's outputs per setting.
    global DATA_ROOT
    if str(args.data_root).strip():
        DATA_ROOT = Path(str(args.data_root)).expanduser().resolve()

    _ensure_clean_dir(DATA_ROOT, force=False)
    query_dir = _query_dir()
    remote_dir = _remote_dir()
    aggregated_json = _aggregated_json()
    aggregated_csv = _aggregated_csv()
    latency_model_json = _latency_model_json()
    provenance_json = _provenance_json()

    _ensure_clean_dir(query_dir, force=bool(args.force))
    _ensure_clean_dir(remote_dir, force=bool(args.force))
    latency_model = _resolve_latency_model(args, latency_model_json)
    _write_provenance(args, latency_model=latency_model, provenance_json=provenance_json, latency_model_json=latency_model_json)

    if not bool(args.remote_export):
        for dataset in datasets:
            for reorder_method, suffix in [("identity", "original"), ("hilbert", "reordered")]:
                out_path = query_dir / f"{dataset}_{suffix}.json"
                if out_path.exists() and not bool(args.force):
                    continue
                if args.skip_export and out_path.exists():
                    continue
                _export_query_ranges(args, dataset, reorder_method, out_path)

    if args.check_only:
        print(f"[plan] aggregated_json={aggregated_json}")
        return 0

    if not args.skip_remote:
        remote_root = _prepare_remote_workspace(
            args,
            Path(args.local_lance_root).resolve(),
            datasets=datasets,
            query_dir=query_dir,
            latency_model_json=latency_model_json,
        )
        _run_remote_repeats(args, datasets, remote_root, remote_dir=remote_dir)
    else:
        _run_local_repeats(
            args,
            datasets,
            Path(args.local_lance_root).resolve(),
            query_dir=query_dir,
            latency_model_json=latency_model_json,
            remote_dir=remote_dir,
        )

    _aggregate_repeat_docs(datasets, remote_dir=remote_dir, aggregated_json=aggregated_json, aggregated_csv=aggregated_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


HERE = Path(__file__).resolve().parent
DEFAULT_S3_BUCKET = os.environ.get("VECIO_S3_BUCKET", "")
DEFAULT_S3_REGION = os.environ.get("VECIO_S3_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def _discover_root() -> Path:
    candidates = [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents]
    for cand in candidates:
        if (cand / "runners" / "shared" / "export_real_query_ranges_for_lance.py").exists():
            return cand.resolve()
    raise RuntimeError("failed to locate experiment repo root from script path or cwd")


ROOT = _discover_root()
DATA_ROOT = ROOT / "experiments" / HERE.name / "data" / "latest"
LANCE_ROOT = Path(os.environ.get("VECIO_LANCE_ROOT", str(ROOT.parent / "lance")))
EXPORT_SCRIPT = ROOT / "runners" / "shared" / "export_real_query_ranges_for_lance.py"
FIG3_FIT_SUMMARY = (
    ROOT
    / "experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves"
    / "data/paper/fit_summary.json"
)

DEFAULTS = {
    "datasets": "sift1m,glove",
    "nlist": 4096,
    "nprobe": 128,
    "num_queries": 120,
    "warmup_queries": 20,
    "query_subset_mode": "random",
    "query_subset_seed": 12345,
    "query_concurrencies": "1,4",
    "io_thread_settings": "8,16,32",
    "environments": "",
    "ks": "16,64",
    "balanced_ps": "1,2,4,8,16,32,64,128,256,512,1024",
    "greedy_gap_kb": "64,128,256,512,1024,2048,4096,8192,16384",
    "greedy_span_caps_mb": "1,2,4,8,16,32",
    "reordered_method": "pca_nd_zorder",
    "repeats": 1,
    "resume_rounds": 100,
    "resume_sleep_seconds": 10,
}

PRESETS = {
    "paper": {},
    "sanity": {
        "datasets": "sift1m",
        "num_queries": 16,
        "warmup_queries": 4,
        "query_concurrencies": "1",
        "io_thread_settings": "16",
        "balanced_ps": "1,4,16,64",
        "greedy_gap_kb": "64,256,1024,4096",
        "greedy_span_caps_mb": "4,8,16",
    },
    "env3": {
        # Paper closed-loop figure covers all six datasets under three env
        # combinations. Keep the full list so `--preset env3` reproduces
        # the figure end-to-end.
        "datasets": "sift1m,glove,deep1m,openimage,msong,random1m128",
        "query_concurrencies": "1,4",
        "io_thread_settings": "8,64",
        "environments": "q1-io64,q4-io64,q4-io8",
        "greedy_gap_kb": "64,256,1024,4096",
        "greedy_span_caps_mb": "4,8,16",
        "repeats": 3,
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run closed-loop P sweeps on real S3.")
    ap.add_argument("--preset", default="paper", choices=sorted(PRESETS))
    ap.add_argument("--datasets", default=DEFAULTS["datasets"])
    ap.add_argument("--nlist", type=int, default=DEFAULTS["nlist"])
    ap.add_argument("--nprobe", type=int, default=DEFAULTS["nprobe"])
    ap.add_argument("--num-queries", type=int, default=DEFAULTS["num_queries"])
    ap.add_argument("--warmup-queries", type=int, default=DEFAULTS["warmup_queries"])
    ap.add_argument("--query-subset-mode", default=DEFAULTS["query_subset_mode"], choices=["random", "first"])
    ap.add_argument("--query-subset-seed", type=int, default=DEFAULTS["query_subset_seed"])
    ap.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="Optional IVF training size for query-range export. 0 means use the exporter default.",
    )
    ap.add_argument("--query-concurrencies", default=DEFAULTS["query_concurrencies"])
    ap.add_argument("--io-thread-settings", default=DEFAULTS["io_thread_settings"])
    ap.add_argument(
        "--environments",
        default="",
        help="Explicit environment pairs such as q1-io64,q4-io64,q4-io8. When set, overrides the cartesian product of --query-concurrencies and --io-thread-settings.",
    )
    ap.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    ap.add_argument("--s3-region", default=DEFAULT_S3_REGION)
    ap.add_argument("--ks", default=DEFAULTS["ks"])
    ap.add_argument("--balanced-ps", default=DEFAULTS["balanced_ps"])
    ap.add_argument("--greedy-gap-kb", default=DEFAULTS["greedy_gap_kb"])
    ap.add_argument("--greedy-span-caps-mb", default=DEFAULTS["greedy_span_caps_mb"])
    ap.add_argument(
        "--reordered-method",
        default=DEFAULTS["reordered_method"],
        choices=[
            "random",
            "pca",
            "pca_zorder",
            "pca_nd_zorder",
            "lpp",
            "tsne",
            "hilbert",
        ],
        help="Reorder method used for canonical *_reordered query ranges.",
    )
    ap.add_argument("--balanced-exact-requests", action="store_true")
    ap.add_argument(
        "--max-iop-size-bytes",
        type=int,
        default=512 * 1024 * 1024 * 1024,
        help="Set very large by default so WaRP does not hit a global request-size cap.",
    )
    ap.add_argument("--repeats", type=int, default=DEFAULTS["repeats"])
    ap.add_argument("--resume-rounds", type=int, default=DEFAULTS["resume_rounds"])
    ap.add_argument("--resume-sleep-seconds", type=int, default=DEFAULTS["resume_sleep_seconds"])
    ap.add_argument("--ssh-target", default="")
    ap.add_argument("--ssh-key", default="")
    ap.add_argument("--remote-work-dir", default="/tmp/vecio_closed_loop_qps_latency_tradeoff")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--local-lance-root", default=str(LANCE_ROOT))
    ap.add_argument("--latency-model-json-in", default="")
    ap.add_argument("--reuse-query-ranges", action="store_true")
    ap.add_argument("--reuse-remote-workspace", action="store_true")
    ap.add_argument("--skip-remote-build", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-remote", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    _apply_preset(args)
    return args


def _apply_preset(args: argparse.Namespace) -> None:
    preset = PRESETS.get(str(args.preset), {})
    if not preset:
        return
    for key, preset_value in preset.items():
        if getattr(args, key) == DEFAULTS[key]:
            setattr(args, key, preset_value)


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


def _parse_env_pairs(raw: str, *, query_concurrencies: str, io_thread_settings: str) -> List[Tuple[int, int]]:
    text = str(raw).strip()
    if not text:
        qs = _parse_int_csv(query_concurrencies)
        ios = _parse_int_csv(io_thread_settings)
        return [(q, io_threads) for q in qs for io_threads in ios]

    out: List[Tuple[int, int]] = []
    for token in [tok.strip().lower() for tok in text.split(",") if tok.strip()]:
        parts = token.split("-")
        q = None
        io_threads = None
        for part in parts:
            if part.startswith("q"):
                q = int(part[1:])
            elif part.startswith("io"):
                io_threads = int(part[2:])
        if q is None or io_threads is None:
            raise ValueError(f"invalid environment spec: {token!r}")
        out.append((q, io_threads))
    if not out:
        raise ValueError("empty environment list")
    return out


def _median(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(statistics.median(xs))


def _run(cmd: List[str], *, cwd: Path | None = None, check_only: bool = False, env: dict | None = None) -> None:
    print("$ " + " ".join(cmd))
    if check_only:
        return
    is_ssh = len(cmd) >= 1 and cmd[0] == "ssh"
    is_remote_benchmark = (
        len(cmd) >= 2
        and cmd[0] == "ssh"
        and any("strict_budget_real_s3" in part for part in cmd[1:])
    )
    is_rsync = len(cmd) >= 1 and cmd[0] == "rsync"
    attempts = 5 if (is_ssh or is_rsync) else 1
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)
            return
        except subprocess.CalledProcessError:
            if attempt >= attempts:
                raise
            if is_remote_benchmark:
                kind = "remote benchmark"
            elif is_rsync:
                kind = "rsync"
            elif is_ssh:
                kind = "ssh"
            else:
                kind = "command"
            print(f"[retry] {kind} attempt {attempt}/{attempts} failed; retrying in 3s")
            time.sleep(3)


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
    if bool(args.reuse_query_ranges) and out_path.exists():
        print(f"[reuse] query ranges: {out_path}")
        return
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
        *(
            ["--train-size", str(int(args.train_size))]
            if int(args.train_size) > 0
            else []
        ),
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


def _export_query_range_pair(
    args: argparse.Namespace,
    dataset: str,
    *,
    original_out_path: Path,
    reordered_out_path: Path,
) -> None:
    if (
        bool(args.reuse_query_ranges)
        and original_out_path.exists()
        and reordered_out_path.exists()
    ):
        print(f"[reuse] query ranges: {original_out_path}")
        print(f"[reuse] query ranges: {reordered_out_path}")
        return

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
        *(
            ["--train-size", str(int(args.train_size))]
            if int(args.train_size) > 0
            else []
        ),
        "--reorder-method",
        "identity",
        "--out",
        str(original_out_path),
        "--extra-reorder-output",
        f"{str(args.reordered_method)}={reordered_out_path}",
        "--s3-bucket",
        str(args.s3_bucket),
        "--s3-region",
        str(args.s3_region),
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


def _canonical_method_name(raw: object) -> str:
    method = str(raw)
    if method == "native_unbounded":
        return "lance_native"
    return method


def _extract_p(method: str) -> float | None:
    if "_p" not in method:
        return None
    tail = method.rsplit("_p", 1)[-1]
    try:
        return float(tail)
    except ValueError:
        return None


def _prepare_remote_workspace(args: argparse.Namespace, local_lance_root: Path, *, query_dir: Path, latency_model_json: Path) -> Path:
    remote_root = str(args.remote_work_dir).rstrip("/")
    remote_lance = f"{remote_root}/lance"
    remote_inputs = f"{remote_root}/inputs"
    if bool(args.reuse_remote_workspace):
        _run(_remote_cmd(args, f"mkdir -p {remote_root} {remote_inputs}"), check_only=bool(args.check_only))
    else:
        _run(_remote_cmd(args, f"mkdir -p {remote_root} {remote_inputs} && rm -rf {remote_lance}"), check_only=bool(args.check_only))
        _rsync_to_remote(
            Path(str(local_lance_root)).resolve().as_posix() + "/",
            f"{args.ssh_target}:{remote_lance}/",
            str(args.ssh_key),
            delete=True,
            excludes=["target", ".git"],
            check_only=bool(args.check_only),
        )
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
    if not bool(args.skip_remote_build):
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
            help_proc = subprocess.run(
                [str(local_bin), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            help_text = help_proc.stdout or ""
            if "--greedy-gap-kb" not in help_text:
                print(
                    "[build] strict_budget_real_s3 binary is stale; rebuilding for greedy-gap-kb CLI",
                    flush=True,
                )
                need_build = True
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
    remote_locks = f"{remote_root.as_posix()}/locks"
    env_pairs = _parse_env_pairs(
        str(args.environments),
        query_concurrencies=str(args.query_concurrencies),
        io_thread_settings=str(args.io_thread_settings),
    )
    _run(_remote_cmd(args, f"mkdir -p {remote_results} {remote_locks}"), check_only=bool(args.check_only))
    remote_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        for query_concurrency, io_threads in env_pairs:
            for repeat_idx in range(1, int(args.repeats) + 1):
                rep = f"repeat_{repeat_idx:02d}"
                stem = f"{dataset}_q{int(query_concurrency)}_io{int(io_threads)}_{rep}"
                out_json = f"{remote_results}/{stem}.json"
                out_csv = f"{remote_results}/{stem}.csv"
                local_json = remote_dir / f"{stem}.json"
                local_csv = remote_dir / f"{stem}.csv"
                if local_json.exists() and local_csv.exists():
                    print(f"[skip] local results already present: {stem}")
                    continue
                lockdir = f"{remote_locks}/{stem}.lock"
                remote_script = " ".join(
                    [
                        f"if [ -f {out_json} ] && [ -f {out_csv} ]; then echo '[skip] remote results already present: {stem}'; exit 0; fi;",
                        f"if ! mkdir {lockdir} 2>/dev/null; then echo '[skip] remote lock already held: {stem}'; exit 0; fi;",
                        f"trap 'rm -rf {lockdir}' EXIT;",
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
                        f"--query-concurrency {int(query_concurrency)}",
                        f"--balanced-ps {args.balanced_ps}",
                        "--balanced-exact-requests" if bool(args.balanced_exact_requests) else "",
                        f"--process-io-threads-limit {int(io_threads)}",
                        f"--cloud-io-parallelism {int(io_threads)}",
                        f"--max-iop-size-bytes {int(args.max_iop_size_bytes)}",
                    ]
                )
                cmd = _remote_cmd(args, remote_script)
                _run(cmd, check_only=bool(args.check_only))
                _rsync_from_remote(
                    f"{args.ssh_target}:{remote_results}/{stem}.json",
                    remote_dir,
                    str(args.ssh_key),
                    delete=False,
                    check_only=bool(args.check_only),
                )
                _rsync_from_remote(
                    f"{args.ssh_target}:{remote_results}/{stem}.csv",
                    remote_dir,
                    str(args.ssh_key),
                    delete=False,
                    check_only=bool(args.check_only),
                )


def _run_local_repeats(
    args: argparse.Namespace,
    datasets: List[str],
    local_lance_root: Path,
    *,
    query_dir: Path,
    latency_model_json: Path,
    remote_dir: Path,
) -> None:
    local_bin = _prepare_local_binary(args, local_lance_root)
    _ensure_clean_dir(remote_dir, force=bool(args.force))
    env_pairs = _parse_env_pairs(
        str(args.environments),
        query_concurrencies=str(args.query_concurrencies),
        io_thread_settings=str(args.io_thread_settings),
    )
    env = dict(os.environ)
    env["AWS_DEFAULT_REGION"] = str(args.s3_region)
    for dataset in datasets:
        for query_concurrency, io_threads in env_pairs:
            for repeat_idx in range(1, int(args.repeats) + 1):
                rep = f"repeat_{repeat_idx:02d}"
                stem = f"{dataset}_q{int(query_concurrency)}_io{int(io_threads)}_{rep}"
                out_json = remote_dir / f"{stem}.json"
                out_csv = remote_dir / f"{stem}.csv"
                if out_json.exists() and out_csv.exists():
                    print(f"[skip] local results already present: {stem}")
                    continue
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
                    str(int(query_concurrency)),
                    "--balanced-ps",
                    str(args.balanced_ps),
                    "--process-io-threads-limit",
                    str(int(io_threads)),
                    "--cloud-io-parallelism",
                    str(int(io_threads)),
                    "--max-iop-size-bytes",
                    str(int(args.max_iop_size_bytes)),
                ]
                if bool(args.balanced_exact_requests):
                    cmd.append("--balanced-exact-requests")
                _run(cmd, check_only=bool(args.check_only), env=env)


def _expected_result_stems(args: argparse.Namespace, datasets: List[str]) -> List[str]:
    env_pairs = _parse_env_pairs(
        str(args.environments),
        query_concurrencies=str(args.query_concurrencies),
        io_thread_settings=str(args.io_thread_settings),
    )
    out: List[str] = []
    for dataset in datasets:
        for query_concurrency, io_threads in env_pairs:
            for repeat_idx in range(1, int(args.repeats) + 1):
                rep = f"repeat_{repeat_idx:02d}"
                out.append(f"{dataset}_q{int(query_concurrency)}_io{int(io_threads)}_{rep}")
    return out


def _missing_local_results(remote_dir: Path, stems: List[str]) -> List[str]:
    missing: List[str] = []
    for stem in stems:
        if not (remote_dir / f"{stem}.json").exists() or not (remote_dir / f"{stem}.csv").exists():
            missing.append(stem)
    return missing


def _run_remote_until_complete(args: argparse.Namespace, datasets: List[str], remote_root: Path, *, remote_dir: Path) -> None:
    expected = _expected_result_stems(args, datasets)
    rounds = max(1, int(args.resume_rounds))
    sleep_seconds = max(1, int(args.resume_sleep_seconds))
    last_missing: List[str] | None = None
    for round_idx in range(1, rounds + 1):
        missing = _missing_local_results(remote_dir, expected)
        if not missing:
            print("[complete] all remote result files are present locally")
            return
        preview = ", ".join(missing[:4])
        if len(missing) > 4:
            preview += ", ..."
        print(f"[resume] round {round_idx}/{rounds}: missing {len(missing)} result sets: {preview}")
        try:
            _run_remote_repeats(args, datasets, remote_root, remote_dir=remote_dir)
        except subprocess.CalledProcessError as exc:
            print(f"[resume] transient remote failure: {exc}. Will retry after {sleep_seconds}s.")
        except Exception as exc:
            print(f"[resume] unexpected remote failure: {exc}. Will retry after {sleep_seconds}s.")
        current_missing = _missing_local_results(remote_dir, expected)
        if not current_missing:
            print("[complete] all remote result files are present locally")
            return
        if last_missing is not None and current_missing == last_missing:
            print(f"[resume] no new local results this round; sleeping {sleep_seconds}s before retry")
        else:
            print(f"[resume] progress made; {len(current_missing)} result sets still missing")
        last_missing = current_missing
        time.sleep(sleep_seconds)
    final_missing = _missing_local_results(remote_dir, expected)
    raise SystemExit(
        f"remote experiment did not complete after {rounds} resume rounds; still missing {len(final_missing)} result sets"
    )


def _choose_greedy_point(rows: List[dict], *, pick: str) -> dict | None:
    full = [r for r in rows if float(r.get("coverage", 0.0)) >= 1.0 - 1e-12]
    feasible = full
    if not feasible:
        return None
    if pick == "qps":
        feasible = [r for r in feasible if r.get("qps") is not None]
        if not feasible:
            return None
        feasible.sort(
            key=lambda r: (
                -float(r.get("coverage", 0.0)),
                -float(r["qps"]),
                float(r.get("avg_latency_ms") or 1e30),
                float(r.get("avg_num_requests") or 1e30),
            )
        )
        return feasible[0]
    feasible = [r for r in feasible if r.get("avg_latency_ms") is not None]
    if not feasible:
        return None
    feasible.sort(
        key=lambda r: (
            -float(r.get("coverage", 0.0)),
            float(r["avg_latency_ms"]),
            float(r.get("p95_latency_ms") or 1e30),
            float(r.get("avg_num_requests") or 1e30),
        )
    )
    return feasible[0]


def _aggregate_repeat_docs(
    datasets: List[str],
    *,
    remote_dir: Path,
    aggregated_json: Path,
    aggregated_csv: Path,
    reordered_method: str,
) -> dict:
    docs = []
    for dataset in datasets:
        per_dataset = []
        for path in sorted(remote_dir.glob(f"{dataset}_q*_io*_repeat_*.json")):
            per_dataset.append(json.loads(path.read_text(encoding="utf-8")))
        if not per_dataset:
            raise SystemExit(f"no remote repeat json for dataset={dataset} under {remote_dir}")
        docs.append((dataset, per_dataset))

    first = docs[0][1][0]
    out = {
        "figure": "figure_closed_loop_qps_latency_tradeoff",
        "nlist": int(first.get("nlist", 4096)),
        "nprobe": int(first.get("nprobe", 128)),
        "layout_methods": {
            "original": "identity",
            "reordered": str(reordered_method),
        },
        "reordered_method": str(reordered_method),
        "query_concurrencies": sorted({int(doc.get("query_concurrency", 1)) for _, items in docs for doc in items}),
        "io_thread_settings": sorted({int(doc.get("process_io_threads_limit", 8)) for _, items in docs for doc in items}),
        "balanced_ps": [float(x) for x in first.get("balanced_ps", [])],
        "repeats": int(min(len(items) for _, items in docs)),
        "rows": [],
        "greedy_candidates": [],
        "greedy_unbounded_candidates": [],
        "selected_points": [],
    }

    for dataset, repeat_docs in docs:
        summary_groups: Dict[Tuple[int, int, int, str, str, int | None], List[dict]] = defaultdict(list)
        greedy_groups: Dict[Tuple[int, int, int, str, int, int, int], List[dict]] = defaultdict(list)
        greedy_unbounded_groups: Dict[Tuple[int, int, int, str, int, int], List[dict]] = defaultdict(list)
        for doc in repeat_docs:
            q = int(doc.get("query_concurrency", 1))
            io_proc = int(doc.get("process_io_threads_limit", 8))
            io_cloud = int(doc.get("cloud_io_parallelism", io_proc))
            for row in doc.get("rows", []):
                summary_groups[(q, io_proc, io_cloud, str(row.get("layout")), _canonical_method_name(row.get("method")), row.get("budget_k"))].append(row)
            for row in doc.get("greedy_candidates", []):
                greedy_groups[
                    (
                        q,
                        io_proc,
                        io_cloud,
                        str(row.get("layout")),
                        int(row.get("budget_k")),
                        int(row.get("gap_kb")),
                        int(row.get("span_cap_bytes")),
                    )
                ].append(row)
            for row in doc.get("greedy_unbounded_candidates", []):
                greedy_unbounded_groups[
                    (
                        q,
                        io_proc,
                        io_cloud,
                        str(row.get("layout")),
                        int(row.get("gap_kb")),
                        int(row.get("span_cap_bytes")),
                    )
                ].append(row)

        aggregated_greedy: List[dict] = []
        for (q, io_proc, io_cloud, layout, budget_k, gap_kb, span_cap_bytes), rows in sorted(greedy_groups.items()):
            aggregated_greedy.append(
                {
                    "dataset": dataset,
                    "query_concurrency": q,
                    "process_io_threads_limit": io_proc,
                    "cloud_io_parallelism": io_cloud,
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
                }
            )
        out["greedy_candidates"].extend(aggregated_greedy)

        aggregated_greedy_unbounded: List[dict] = []
        for (q, io_proc, io_cloud, layout, gap_kb, span_cap_bytes), rows in sorted(greedy_unbounded_groups.items()):
            aggregated_greedy_unbounded.append(
                {
                    "dataset": dataset,
                    "query_concurrency": q,
                    "process_io_threads_limit": io_proc,
                    "cloud_io_parallelism": io_cloud,
                    "layout": layout,
                    "gap_kb": int(gap_kb),
                    "span_cap_bytes": int(span_cap_bytes),
                    "wall_ms": _median(r.get("wall_ms") for r in rows),
                    "qps": _median(r.get("qps") for r in rows),
                    "avg_latency_ms": _median(r.get("avg_latency_ms") for r in rows),
                    "p95_latency_ms": _median(r.get("p95_latency_ms") for r in rows),
                    "avg_num_requests": _median(r.get("avg_num_requests") for r in rows),
                    "avg_planned_bytes": _median(r.get("avg_planned_bytes") for r in rows),
                    "avg_model_work_ms": _median(r.get("avg_model_work_ms") for r in rows),
                    "avg_model_bottleneck_ms": _median(r.get("avg_model_bottleneck_ms") for r in rows),
                    "avg_model_objective_ms": _median(r.get("avg_model_objective_ms") for r in rows),
                }
            )
        out["greedy_unbounded_candidates"].extend(aggregated_greedy_unbounded)

        for (q, io_proc, io_cloud, layout, method, budget_k), rows in sorted(summary_groups.items()):
            if method == "greedy_gap_frontier":
                continue
            out["rows"].append(
                {
                    "dataset": dataset,
                    "query_concurrency": q,
                    "process_io_threads_limit": io_proc,
                    "cloud_io_parallelism": io_cloud,
                    "layout": layout,
                    "method": method,
                    "budget_k": budget_k,
                    "p": _extract_p(method),
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
                }
            )

        selected_groups: Dict[Tuple[int, int, int], List[dict]] = defaultdict(list)
        for row in aggregated_greedy:
            selected_groups[(int(row["query_concurrency"]), int(row["process_io_threads_limit"]), int(row["cloud_io_parallelism"]))].append(row)

        for (q, io_proc, io_cloud), group_rows in sorted(selected_groups.items()):
            for method_name, chosen in [
                ("greedy_best_latency", _choose_greedy_point(group_rows, pick="latency")),
                ("greedy_best_qps", _choose_greedy_point(group_rows, pick="qps")),
            ]:
                if chosen is None:
                    continue
                out["selected_points"].append(
                    {
                        "dataset": dataset,
                        "query_concurrency": q,
                        "process_io_threads_limit": io_proc,
                        "cloud_io_parallelism": io_cloud,
                        "layout": chosen["layout"],
                        "method": method_name,
                        "budget_k": int(chosen["budget_k"]),
                        "gap_kb": int(chosen["gap_kb"]),
                        "span_cap_bytes": int(chosen["span_cap_bytes"]),
                        "coverage": float(chosen["coverage"]),
                        "wall_ms": chosen["wall_ms"],
                        "qps": chosen["qps"],
                        "avg_latency_ms": chosen["avg_latency_ms"],
                        "p95_latency_ms": chosen["p95_latency_ms"],
                        "avg_num_requests": chosen["avg_num_requests"],
                        "avg_planned_bytes": chosen["avg_planned_bytes"],
                        "avg_model_work_ms": chosen["avg_model_work_ms"],
                        "avg_model_bottleneck_ms": chosen["avg_model_bottleneck_ms"],
                        "avg_model_objective_ms": chosen["avg_model_objective_ms"],
                    }
                )

    aggregated_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    lines = [
        "section,dataset,layout,method,budget_k,p,status,coverage,wall_ms,qps,avg_latency_ms,p95_latency_ms,avg_num_requests,avg_planned_bytes,avg_model_work_ms,avg_model_bottleneck_ms,avg_model_objective_ms,gap_kb,span_cap_bytes"
    ]
    for row in out["rows"]:
        lines.append(
            "summary,{dataset},{layout},{method},{budget_k},{p},{status},{coverage:.6f},{wall_ms},{qps},{avg_latency_ms},{p95_latency_ms},{avg_num_requests},{avg_planned_bytes},{avg_model_work_ms},{avg_model_bottleneck_ms},{avg_model_objective_ms},,".format(
                dataset=row["dataset"],
                layout=row["layout"],
                method=row["method"],
                budget_k="" if row["budget_k"] is None else row["budget_k"],
                p="" if row["p"] is None else row["p"],
                status=row["status"],
                coverage=float(row["coverage"]),
                wall_ms="" if row["wall_ms"] is None else f"{float(row['wall_ms']):.6f}",
                qps="" if row["qps"] is None else f"{float(row['qps']):.6f}",
                avg_latency_ms="" if row["avg_latency_ms"] is None else f"{float(row['avg_latency_ms']):.6f}",
                p95_latency_ms="" if row["p95_latency_ms"] is None else f"{float(row['p95_latency_ms']):.6f}",
                avg_num_requests="" if row["avg_num_requests"] is None else f"{float(row['avg_num_requests']):.6f}",
                avg_planned_bytes="" if row["avg_planned_bytes"] is None else f"{float(row['avg_planned_bytes']):.6f}",
                avg_model_work_ms="" if row["avg_model_work_ms"] is None else f"{float(row['avg_model_work_ms']):.6f}",
                avg_model_bottleneck_ms="" if row["avg_model_bottleneck_ms"] is None else f"{float(row['avg_model_bottleneck_ms']):.6f}",
                avg_model_objective_ms="" if row["avg_model_objective_ms"] is None else f"{float(row['avg_model_objective_ms']):.6f}",
            )
        )
    for row in out["greedy_candidates"]:
        lines.append(
            "greedy,{dataset},{layout},greedy_candidate,{budget_k},,ok,{coverage:.6f},{wall_ms},{qps},{avg_latency_ms},{p95_latency_ms},{avg_num_requests},{avg_planned_bytes},{avg_model_work_ms},{avg_model_bottleneck_ms},{avg_model_objective_ms},{gap_kb},{span_cap_bytes}".format(
                dataset=row["dataset"],
                layout=row["layout"],
                budget_k=int(row["budget_k"]),
                coverage=float(row["coverage"]),
                wall_ms="" if row["wall_ms"] is None else f"{float(row['wall_ms']):.6f}",
                qps="" if row["qps"] is None else f"{float(row['qps']):.6f}",
                avg_latency_ms="" if row["avg_latency_ms"] is None else f"{float(row['avg_latency_ms']):.6f}",
                p95_latency_ms="" if row["p95_latency_ms"] is None else f"{float(row['p95_latency_ms']):.6f}",
                avg_num_requests="" if row["avg_num_requests"] is None else f"{float(row['avg_num_requests']):.6f}",
                avg_planned_bytes="" if row["avg_planned_bytes"] is None else f"{float(row['avg_planned_bytes']):.6f}",
                avg_model_work_ms="" if row["avg_model_work_ms"] is None else f"{float(row['avg_model_work_ms']):.6f}",
                avg_model_bottleneck_ms="" if row["avg_model_bottleneck_ms"] is None else f"{float(row['avg_model_bottleneck_ms']):.6f}",
                avg_model_objective_ms="" if row["avg_model_objective_ms"] is None else f"{float(row['avg_model_objective_ms']):.6f}",
                gap_kb=int(row["gap_kb"]),
                span_cap_bytes=int(row["span_cap_bytes"]),
            )
        )
    for row in out["greedy_unbounded_candidates"]:
        lines.append(
            "greedy_unbounded,{dataset},{layout},greedy_gap_frontier_unbounded,,,1.000000,{wall_ms},{qps},{avg_latency_ms},{p95_latency_ms},{avg_num_requests},{avg_planned_bytes},{avg_model_work_ms},{avg_model_bottleneck_ms},{avg_model_objective_ms},{gap_kb},{span_cap_bytes}".format(
                dataset=row["dataset"],
                layout=row["layout"],
                wall_ms="" if row["wall_ms"] is None else f"{float(row['wall_ms']):.6f}",
                qps="" if row["qps"] is None else f"{float(row['qps']):.6f}",
                avg_latency_ms="" if row["avg_latency_ms"] is None else f"{float(row['avg_latency_ms']):.6f}",
                p95_latency_ms="" if row["p95_latency_ms"] is None else f"{float(row['p95_latency_ms']):.6f}",
                avg_num_requests="" if row["avg_num_requests"] is None else f"{float(row['avg_num_requests']):.6f}",
                avg_planned_bytes="" if row["avg_planned_bytes"] is None else f"{float(row['avg_planned_bytes']):.6f}",
                avg_model_work_ms="" if row["avg_model_work_ms"] is None else f"{float(row['avg_model_work_ms']):.6f}",
                avg_model_bottleneck_ms="" if row["avg_model_bottleneck_ms"] is None else f"{float(row['avg_model_bottleneck_ms']):.6f}",
                avg_model_objective_ms="" if row["avg_model_objective_ms"] is None else f"{float(row['avg_model_objective_ms']):.6f}",
                gap_kb=int(row["gap_kb"]),
                span_cap_bytes=int(row["span_cap_bytes"]),
            )
        )
    for row in out["selected_points"]:
        lines.append(
            "selected,{dataset},{layout},{method},{budget_k},,ok,{coverage:.6f},{wall_ms},{qps},{avg_latency_ms},{p95_latency_ms},{avg_num_requests},{avg_planned_bytes},{avg_model_work_ms},{avg_model_bottleneck_ms},{avg_model_objective_ms},{gap_kb},{span_cap_bytes}".format(
                dataset=row["dataset"],
                layout=row["layout"],
                method=row["method"],
                budget_k=int(row["budget_k"]),
                coverage=float(row["coverage"]),
                wall_ms="" if row["wall_ms"] is None else f"{float(row['wall_ms']):.6f}",
                qps="" if row["qps"] is None else f"{float(row['qps']):.6f}",
                avg_latency_ms="" if row["avg_latency_ms"] is None else f"{float(row['avg_latency_ms']):.6f}",
                p95_latency_ms="" if row["p95_latency_ms"] is None else f"{float(row['p95_latency_ms']):.6f}",
                avg_num_requests="" if row["avg_num_requests"] is None else f"{float(row['avg_num_requests']):.6f}",
                avg_planned_bytes="" if row["avg_planned_bytes"] is None else f"{float(row['avg_planned_bytes']):.6f}",
                avg_model_work_ms="" if row["avg_model_work_ms"] is None else f"{float(row['avg_model_work_ms']):.6f}",
                avg_model_bottleneck_ms="" if row["avg_model_bottleneck_ms"] is None else f"{float(row['avg_model_bottleneck_ms']):.6f}",
                avg_model_objective_ms="" if row["avg_model_objective_ms"] is None else f"{float(row['avg_model_objective_ms']):.6f}",
                gap_kb=int(row["gap_kb"]),
                span_cap_bytes=int(row["span_cap_bytes"]),
            )
        )
    aggregated_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_provenance(args: argparse.Namespace, latency_model: dict, provenance_json: Path, latency_model_json: Path) -> None:
    try:
        latency_model_json_ref = str(latency_model_json.relative_to(HERE))
    except ValueError:
        latency_model_json_ref = str(latency_model_json)
    payload = {
        "figure": "figure_closed_loop_qps_latency_tradeoff",
        "datasets": _parse_csv_list(args.datasets),
        "nlist": int(args.nlist),
        "nprobe": int(args.nprobe),
        "num_queries": int(args.num_queries),
        "warmup_queries": int(args.warmup_queries),
        "query_concurrencies": _parse_int_csv(args.query_concurrencies),
        "environments": [
            {"query_concurrency": int(q), "io_threads": int(io_threads)}
            for q, io_threads in _parse_env_pairs(
                str(args.environments),
                query_concurrencies=str(args.query_concurrencies),
                io_thread_settings=str(args.io_thread_settings),
            )
        ],
        "query_subset_mode": str(args.query_subset_mode),
        "query_subset_seed": int(args.query_subset_seed),
        "train_size": int(args.train_size),
        "layout_methods": {
            "original": "identity",
            "reordered": str(args.reordered_method),
        },
        "reordered_method": str(args.reordered_method),
        "s3_bucket": str(args.s3_bucket),
        "s3_region": str(args.s3_region),
        "ks": _parse_int_csv(args.ks),
        "balanced_ps": _parse_float_csv(args.balanced_ps),
        "balanced_exact_requests": bool(args.balanced_exact_requests),
        "greedy_gap_kb": _parse_int_csv(args.greedy_gap_kb),
        "greedy_span_caps_mb": _parse_int_csv(args.greedy_span_caps_mb),
        "io_thread_settings": _parse_int_csv(args.io_thread_settings),
        "max_iop_size_bytes": int(args.max_iop_size_bytes),
        "repeats": int(args.repeats),
        "preset": str(args.preset),
        "reuse_query_ranges": bool(args.reuse_query_ranges),
        "reuse_remote_workspace": bool(args.reuse_remote_workspace),
        "skip_remote_build": bool(args.skip_remote_build),
        "ssh_target": str(args.ssh_target),
        "latency_model_json": latency_model_json_ref,
        "latency_model": latency_model,
    }
    provenance_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = [d.lower() for d in _parse_csv_list(args.datasets)]
    data_root = (
        Path(str(args.data_root)).expanduser().resolve()
        if str(args.data_root).strip()
        else DATA_ROOT
    )
    # Default to local execution on the current host. Pass --ssh-target to
    # delegate the Lance benchmark to a separate EC2 box (production mode).
    if not args.ssh_target and not args.skip_remote:
        args.skip_remote = True
    if not args.skip_export and not args.check_only and not str(args.s3_bucket).strip():
        raise SystemExit("--s3-bucket is required unless using --skip-export or --check-only")

    if args.check_only:
        print(f"[plan] preset={args.preset}")
        print(f"[plan] datasets={datasets}")
        print(f"[plan] nlist={args.nlist} nprobe={args.nprobe}")
        print(f"[plan] reordered_method={args.reordered_method}")
        print(f"[plan] query_concurrencies={args.query_concurrencies}")
        print(f"[plan] io_thread_settings={args.io_thread_settings}")
        print(f"[plan] environments={args.environments or '(cartesian product)'}")
        print(f"[plan] ks={args.ks}")
        print(f"[plan] balanced_ps={args.balanced_ps}")
        print(f"[plan] reuse_query_ranges={bool(args.reuse_query_ranges)}")
        print(f"[plan] reuse_remote_workspace={bool(args.reuse_remote_workspace)}")
        print(f"[plan] skip_remote_build={bool(args.skip_remote_build)}")
        print(f"[plan] max_iop_size_bytes={args.max_iop_size_bytes}")
        print("[plan] no export / no remote run / no local data writes in --check-only mode")
        return 0

    _ensure_clean_dir(data_root, force=False)
    query_dir = data_root / "query_ranges"
    remote_dir = data_root / "remote_runs"
    aggregated_json = data_root / "aggregated_results.json"
    aggregated_csv = data_root / "aggregated_results.csv"
    latency_model_json = data_root / "latency_model.json"
    provenance_json = data_root / "data_provenance.json"

    _ensure_clean_dir(query_dir, force=bool(args.force))
    _ensure_clean_dir(remote_dir, force=bool(args.force))
    latency_model = _resolve_latency_model(args, latency_model_json)
    _write_provenance(args, latency_model=latency_model, provenance_json=provenance_json, latency_model_json=latency_model_json)

    if not args.skip_export:
        for dataset in datasets:
            _export_query_range_pair(
                args,
                dataset,
                original_out_path=query_dir / f"{dataset}_original.json",
                reordered_out_path=query_dir / f"{dataset}_reordered.json",
            )

    if not args.skip_remote:
        remote_root = _prepare_remote_workspace(args, Path(str(args.local_lance_root)).resolve(), query_dir=query_dir, latency_model_json=latency_model_json)
        _run_remote_until_complete(args, datasets, remote_root, remote_dir=remote_dir)
    else:
        _run_local_repeats(
            args,
            datasets,
            Path(str(args.local_lance_root)).resolve(),
            query_dir=query_dir,
            latency_model_json=latency_model_json,
            remote_dir=remote_dir,
        )

    _aggregate_repeat_docs(
        datasets,
        remote_dir=remote_dir,
        aggregated_json=aggregated_json,
        aggregated_csv=aggregated_csv,
        reordered_method=str(args.reordered_method),
    )

    # Post-process: publish the aggregated JSON under the canonical name
    # and subdirectory that the experiment-owned renderer reads
    # (data/latest/data_gap_env3_v3/figure_closed_loop_env3_reorder_only_data.json).
    import shutil as _shutil
    canonical_subdir = data_root / "data_gap_env3_v3"
    canonical_subdir.mkdir(parents=True, exist_ok=True)
    canonical_json = canonical_subdir / "figure_closed_loop_env3_reorder_only_data.json"
    _shutil.copy2(aggregated_json, canonical_json)
    print(f"[ok] published canonical render input: {canonical_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

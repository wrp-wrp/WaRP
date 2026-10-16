#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import get_dataset
from src.index_builder import IVFIndexBuilder


HERE = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "experiments" / "figure-14-sift1m-sensitivity" / "data" / "latest"
SETTINGS_ROOT = DATA_ROOT / "settings"
RECALL_CACHE_ROOT = DATA_ROOT / "recall_cache"
STRESS_RUNNER = ROOT / "runners" / "figure-14-sift1m-sensitivity" / "stress_runner.py"
STRESS_DATA_ROOT = DATA_ROOT / "stress"
RUNTIME_DATA_ROOT = DATA_ROOT
RUNTIME_SETTINGS_ROOT = SETTINGS_ROOT
RUNTIME_RECALL_CACHE_ROOT = RECALL_CACHE_ROOT
RUNTIME_STRESS_DATA_ROOT = STRESS_DATA_ROOT


SETTING_SPECS: List[dict] = [
    {
        "setting_id": "nprobe_np32_k8",
        "panel": "nprobe",
        "setting_label": "nprobe=32",
        "dataset": "sift1m",
        "nlist": 4096,
        "nprobe": 32,
        "budget_ks": [8, 16],
    },
    {
        "setting_id": "nprobe_np64_k16",
        "panel": "nprobe",
        "setting_label": "nprobe=64",
        "dataset": "sift1m",
        "nlist": 4096,
        "nprobe": 64,
        "budget_ks": [16, 32],
    },
    {
        "setting_id": "nprobe_np128_k32",
        "panel": "nprobe",
        "setting_label": "nprobe=128",
        "dataset": "sift1m",
        "nlist": 4096,
        "nprobe": 128,
        "budget_ks": [32, 64],
    },
    {
        "setting_id": "nprobe_np256_k64",
        "panel": "nprobe",
        "setting_label": "nprobe=256",
        "dataset": "sift1m",
        "nlist": 4096,
        "nprobe": 256,
        "budget_ks": [64, 128],
    },
    {
        "setting_id": "nprobe_np512_k128",
        "panel": "nprobe",
        "setting_label": "nprobe=512",
        "dataset": "sift1m",
        "nlist": 4096,
        "nprobe": 512,
        "budget_ks": [128, 256],
    },
    {
        "setting_id": "cluster_nlist1024_np32_k8",
        "panel": "cluster",
        "setting_label": "nlist=1024, nprobe=32",
        "dataset": "sift1m",
        "nlist": 1024,
        "nprobe": 32,
        "budget_ks": [8, 16],
    },
    {
        "setting_id": "cluster_nlist2048_np64_k16",
        "panel": "cluster",
        "setting_label": "nlist=2048, nprobe=64",
        "dataset": "sift1m",
        "nlist": 2048,
        "nprobe": 64,
        "budget_ks": [16, 32],
    },
    {
        "setting_id": "cluster_nlist4096_np128_k32",
        "panel": "cluster",
        "setting_label": "nlist=4096, nprobe=128",
        "dataset": "sift1m",
        "nlist": 4096,
        "nprobe": 128,
        "budget_ks": [32, 64],
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the SIFT1M real-S3 boundary workflow.")
    ap.add_argument("--settings", default="", help="Comma-separated setting ids; default runs all.")
    ap.add_argument("--ssh-target", default="")
    ap.add_argument("--ssh-key", default="")
    ap.add_argument("--balanced-ps", default="16")
    ap.add_argument("--greedy-gap-kb", default="64,128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--greedy-span-caps-mb", default="1,2,4,8,16,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-queries", type=int, default=240)
    ap.add_argument("--warmup-queries", type=int, default=20)
    ap.add_argument("--query-subset-mode", default="random", choices=["random", "first"])
    ap.add_argument("--query-subset-seed", type=int, default=12345)
    ap.add_argument("--query-concurrency", type=int, default=4)
    ap.add_argument("--max-iop-size-bytes", type=int, default=(1 << 60))
    ap.add_argument("--process-io-threads-limit", type=int, default=64)
    ap.add_argument("--cloud-io-parallelism", type=int, default=64)
    ap.add_argument("--train-size", type=int, default=200000)
    ap.add_argument("--faiss-threads", type=int, default=16)
    ap.add_argument("--remote-work-dir", default="/tmp/warp_sift1m_real_s3_sensitivity")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--stress-data-root", default="")
    ap.add_argument("--remote-export", action="store_true")
    ap.add_argument("--skip-remote-build-after-first", action="store_true", default=True)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _set_runtime_paths(*, data_root: Path, stress_data_root: Path) -> None:
    global RUNTIME_DATA_ROOT
    global RUNTIME_SETTINGS_ROOT
    global RUNTIME_RECALL_CACHE_ROOT
    global RUNTIME_STRESS_DATA_ROOT
    RUNTIME_DATA_ROOT = data_root.resolve()
    RUNTIME_SETTINGS_ROOT = RUNTIME_DATA_ROOT / "settings"
    RUNTIME_RECALL_CACHE_ROOT = RUNTIME_DATA_ROOT / "recall_cache"
    RUNTIME_STRESS_DATA_ROOT = stress_data_root.resolve()


def _run(cmd: List[str], *, cwd: Path | None = None, check_only: bool = False) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if check_only:
        return
    env = os.environ.copy()
    faiss_threads = str(env.get("VECIO_FAISS_THREADS", "")).strip() or str(
        max(1, int(os.environ.get("VECIO_FAISS_THREADS_DEFAULT", "16")))
    )
    env["VECIO_FAISS_THREADS"] = faiss_threads
    env["OMP_NUM_THREADS"] = faiss_threads
    env["OPENBLAS_NUM_THREADS"] = faiss_threads
    env["MKL_NUM_THREADS"] = faiss_threads
    env["NUMEXPR_NUM_THREADS"] = faiss_threads
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _run_capture(cmd: List[str], *, cwd: Path | None = None) -> str:
    print("$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    faiss_threads = str(env.get("VECIO_FAISS_THREADS", "")).strip() or str(
        max(1, int(os.environ.get("VECIO_FAISS_THREADS_DEFAULT", "16")))
    )
    env["VECIO_FAISS_THREADS"] = faiss_threads
    env["OMP_NUM_THREADS"] = faiss_threads
    env["OPENBLAS_NUM_THREADS"] = faiss_threads
    env["MKL_NUM_THREADS"] = faiss_threads
    env["NUMEXPR_NUM_THREADS"] = faiss_threads
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, env=env, text=True)


def _ssh_prefix(ssh_target: str, ssh_key: str) -> List[str]:
    cmd = ["ssh"]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    cmd.extend(["-o", "StrictHostKeyChecking=accept-new", ssh_target])
    return cmd


def _subset_queries(queries: np.ndarray, num_queries: int, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_queries <= 0 or num_queries >= len(queries):
        idx = np.arange(len(queries), dtype=np.int64)
        return queries, idx
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(len(queries), size=int(num_queries), replace=False))
        return queries[idx], idx
    idx = np.arange(int(num_queries), dtype=np.int64)
    return queries[:num_queries], idx


def _recall_cache_path(dataset: str, *, num_queries: int, mode: str, seed: int, k: int) -> Path:
    safe_mode = str(mode).strip().lower()
    return RUNTIME_RECALL_CACHE_ROOT / f"{dataset}_q{int(num_queries)}_{safe_mode}_seed{int(seed)}_k{k}.json"


def _exact_topk_l2_chunked(base: np.ndarray, queries: np.ndarray, *, k: int, chunk_rows: int = 100_000) -> np.ndarray:
    num_queries = int(queries.shape[0])
    best_dists = np.full((num_queries, int(k)), np.inf, dtype=np.float32)
    best_ids = np.full((num_queries, int(k)), -1, dtype=np.int64)
    q = np.ascontiguousarray(queries.astype("float32"))
    q_norm = np.sum(q * q, axis=1, dtype=np.float32)[:, None]
    total = int(base.shape[0])

    for start in range(0, total, int(chunk_rows)):
        end = min(total, start + int(chunk_rows))
        xb = np.ascontiguousarray(np.asarray(base[start:end], dtype=np.float32))
        xb_norm = np.sum(xb * xb, axis=1, dtype=np.float32)[None, :]
        dists = q_norm + xb_norm - 2.0 * (q @ xb.T)
        local_k = min(int(k), int(xb.shape[0]))
        local_idx = np.argpartition(dists, kth=local_k - 1, axis=1)[:, :local_k]
        local_dists = np.take_along_axis(dists, local_idx, axis=1)
        local_ids = local_idx.astype(np.int64) + int(start)

        merged_dists = np.concatenate([best_dists, local_dists.astype(np.float32)], axis=1)
        merged_ids = np.concatenate([best_ids, local_ids], axis=1)
        pick = np.argpartition(merged_dists, kth=int(k) - 1, axis=1)[:, : int(k)]
        best_dists = np.take_along_axis(merged_dists, pick, axis=1)
        best_ids = np.take_along_axis(merged_ids, pick, axis=1)

    order = np.argsort(best_dists, axis=1)
    return np.take_along_axis(best_ids, order, axis=1)


def _load_or_compute_gt_subset(ds, *, dataset: str, base: np.ndarray, query_idx: np.ndarray, selected_queries: np.ndarray, k: int, mode: str, seed: int) -> np.ndarray:
    gt_I, _gt_D = ds.get_groundtruth(k=int(k))
    if gt_I is not None:
        return np.asarray(gt_I, dtype=np.int64)[query_idx][:, : int(k)]

    RUNTIME_RECALL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = _recall_cache_path(dataset, num_queries=len(selected_queries), mode=mode, seed=seed, k=k)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_idx = np.asarray(payload.get("query_indices", []), dtype=np.int64)
        if np.array_equal(cached_idx, np.asarray(query_idx, dtype=np.int64)):
            return np.asarray(payload.get("gt_ids", []), dtype=np.int64)

    print(f"[recall] groundtruth missing for dataset={dataset}; computing exact top-{int(k)} on selected queries", flush=True)
    gt_subset = _exact_topk_l2_chunked(base, selected_queries, k=int(k))
    cache_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "k": int(k),
                "query_indices": [int(x) for x in np.asarray(query_idx).tolist()],
                "gt_ids": [[int(x) for x in row] for row in np.asarray(gt_subset).tolist()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return np.asarray(gt_subset, dtype=np.int64)


def _compute_recall_at_10(spec: dict, *, num_queries: int, mode: str, seed: int, train_size: int) -> dict:
    ds = get_dataset(str(spec["dataset"]))
    ds.prepare()
    base = np.ascontiguousarray(ds.get_base_vectors().astype("float32"))
    queries_all = np.ascontiguousarray(ds.get_queries().astype("float32"))
    selected_queries, query_idx = _subset_queries(queries_all, int(num_queries), mode, int(seed))
    gt_I = _load_or_compute_gt_subset(
        ds,
        dataset=str(spec["dataset"]),
        base=base,
        query_idx=query_idx,
        selected_queries=selected_queries,
        k=10,
        mode=mode,
        seed=seed,
    )
    builder = IVFIndexBuilder(dimension=int(base.shape[1]), num_clusters=int(spec["nlist"]))
    effective_train_size = int(train_size) if int(train_size) > 0 else None
    builder.build(base, train_size=effective_train_size)
    _dist, pred = builder.search(selected_queries, k=10, nprobe=int(spec["nprobe"]))
    hits = 0
    total = 0
    for gt_row, pred_row in zip(gt_I, pred):
        gt_set = {int(x) for x in np.asarray(gt_row).tolist() if int(x) >= 0}
        pred_set = {int(x) for x in np.asarray(pred_row).tolist() if int(x) >= 0}
        hits += len(gt_set & pred_set)
        total += len(gt_set)
    recall = float(hits) / float(total) if total > 0 else 0.0
    return {
        "setting_id": str(spec["setting_id"]),
        "dataset": str(spec["dataset"]),
        "nlist": int(spec["nlist"]),
        "nprobe": int(spec["nprobe"]),
        "budget_ks": [int(v) for v in spec["budget_ks"]],
        "train_size": int(train_size),
        "num_queries": int(len(selected_queries)),
        "query_indices": [int(x) for x in np.asarray(query_idx).tolist()],
        "recall_at_10": recall,
    }


def _compute_recall_at_10_remote(args: argparse.Namespace, spec: dict) -> dict:
    remote_root = str(args.remote_work_dir).rstrip("/")
    remote_vecio = f"{remote_root}/vecio"
    remote_out = f"{remote_root}/inputs/recall_{spec['setting_id']}.json"
    py_stmt = (
        "PY=python3; "
        f"export PYTHONPATH={remote_vecio}:$PYTHONPATH; "
        f"cd {remote_vecio} && "
        '"$PY" runners/figure-14-sift1m-sensitivity/compute_ivf_recall.py '
        f"--dataset {spec['dataset']} "
        f"--nlist {int(spec['nlist'])} "
        f"--nprobe {int(spec['nprobe'])} "
        f"--num-queries {int(args.num_queries)} "
        f"--query-subset-mode {args.query_subset_mode} "
        f"--query-subset-seed {int(args.query_subset_seed)} "
        f"--train-size {int(args.train_size)} "
        f"--out {remote_out}"
    )
    _run(_ssh_prefix(str(args.ssh_target), str(args.ssh_key)) + [py_stmt], check_only=False)
    raw = _run_capture(_ssh_prefix(str(args.ssh_target), str(args.ssh_key)) + [f"cat {remote_out}"])
    return json.loads(raw)


def _write_manifest(args: argparse.Namespace, specs: List[dict]) -> None:
    RUNTIME_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_SETTINGS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "table_sift1m_real_s3_sensitivity",
        "dataset": "sift1m",
        "balanced_ps": [float(tok.strip()) for tok in str(args.balanced_ps).split(",") if tok.strip()],
        "num_queries": int(args.num_queries),
        "warmup_queries": int(args.warmup_queries),
        "query_concurrency": int(args.query_concurrency),
        "query_subset_mode": str(args.query_subset_mode),
        "query_subset_seed": int(args.query_subset_seed),
        "max_iop_size_bytes": int(args.max_iop_size_bytes),
        "settings": specs,
    }
    (RUNTIME_DATA_ROOT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (RUNTIME_DATA_ROOT / "experiment_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setting_id", "panel", "setting_label", "dataset", "nlist", "nprobe", "budget_ks"],
        )
        writer.writeheader()
        for row in specs:
            writer.writerow(
                {
                    "setting_id": row["setting_id"],
                    "panel": row["panel"],
                    "setting_label": row["setting_label"],
                    "dataset": row["dataset"],
                    "nlist": row["nlist"],
                    "nprobe": row["nprobe"],
                    "budget_ks": ",".join(str(int(v)) for v in row["budget_ks"]),
                }
            )


def _selected_specs(raw: str) -> List[dict]:
    if not str(raw).strip():
        return list(SETTING_SPECS)
    requested = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    by_id = {str(spec["setting_id"]): spec for spec in SETTING_SPECS}
    missing = [sid for sid in requested if sid not in by_id]
    if missing:
        raise SystemExit(f"unknown setting ids: {missing}")
    return [by_id[sid] for sid in requested]


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _is_setting_complete(spec: dict, *, repeats: int) -> bool:
    setting_root = RUNTIME_SETTINGS_ROOT / str(spec["setting_id"])
    remote_runs = setting_root / "remote_runs"
    aggregated_json = setting_root / "aggregated_results.json"
    summary_json = setting_root / "setting_summary.json"
    if not remote_runs.exists() or not aggregated_json.exists() or not summary_json.exists():
        return False
    json_count = len(list(remote_runs.glob("*.json")))
    csv_count = len(list(remote_runs.glob("*.csv")))
    return json_count >= int(repeats) and csv_count >= int(repeats)


def _archive_setting_outputs(spec: dict, recall_payload: dict) -> None:
    src_root = RUNTIME_STRESS_DATA_ROOT
    dst_root = RUNTIME_SETTINGS_ROOT / str(spec["setting_id"])
    dst_root.mkdir(parents=True, exist_ok=True)
    for name in ["aggregated_results.json", "aggregated_results.csv", "latency_model.json", "data_provenance.json"]:
        src = src_root / name
        if src.exists():
            shutil.copy2(src, dst_root / name)
    for subdir in ["remote_runs", "query_ranges"]:
        src_dir = src_root / subdir
        if src_dir.exists():
            _copy_tree(src_dir, dst_root / subdir)
    summary = dict(spec)
    summary.update(recall_payload)
    (dst_root / "setting_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _build_runner_cmd(args: argparse.Namespace, spec: dict, *, reuse_remote_workspace: bool) -> List[str]:
    cmd = [
        sys.executable,
        str(STRESS_RUNNER),
        "--datasets",
        str(spec["dataset"]),
        "--nprobe",
        str(int(spec["nprobe"])),
        "--nlist",
        str(int(spec["nlist"])),
        "--num-queries",
        str(int(args.num_queries)),
        "--warmup-queries",
        str(int(args.warmup_queries)),
        "--query-subset-mode",
        str(args.query_subset_mode),
        "--query-subset-seed",
        str(int(args.query_subset_seed)),
        "--query-concurrency",
        str(int(args.query_concurrency)),
        "--ks",
        ",".join(str(int(v)) for v in spec["budget_ks"]),
        "--balanced-ps",
        str(args.balanced_ps),
        "--greedy-gap-kb",
        str(args.greedy_gap_kb),
        "--greedy-span-caps-mb",
        str(args.greedy_span_caps_mb),
        "--repeats",
        str(int(args.repeats)),
        "--train-size",
        str(int(args.train_size)),
        "--max-iop-size-bytes",
        str(int(args.max_iop_size_bytes)),
        "--process-io-threads-limit",
        str(int(args.process_io_threads_limit)),
        "--cloud-io-parallelism",
        str(int(args.cloud_io_parallelism)),
        "--remote-work-dir",
        str(args.remote_work_dir),
        "--data-root",
        str(RUNTIME_STRESS_DATA_ROOT),
    ]
    if bool(args.remote_export):
        cmd.append("--remote-export")
    if reuse_remote_workspace:
        cmd.append("--reuse-remote-workspace")
        if bool(args.skip_remote_build_after_first):
            cmd.append("--skip-remote-build")
    if str(args.ssh_target).strip():
        cmd.extend(["--ssh-target", str(args.ssh_target).strip()])
    if str(args.ssh_key).strip():
        cmd.extend(["--ssh-key", str(args.ssh_key).strip()])
    if bool(args.force):
        cmd.append("--force")
    return cmd


def _run_one_setting(args: argparse.Namespace, spec: dict, *, reuse_remote_workspace: bool) -> None:
    if not args.check_only:
        cmd = _build_runner_cmd(args, spec, reuse_remote_workspace=reuse_remote_workspace)
        _run(cmd, cwd=ROOT, check_only=False)
        if bool(args.remote_export):
            recall_payload = _compute_recall_at_10_remote(args, spec)
        else:
            recall_payload = _compute_recall_at_10(
                spec,
                num_queries=int(args.num_queries),
                mode=str(args.query_subset_mode),
                seed=int(args.query_subset_seed),
                train_size=int(args.train_size),
            )
        _archive_setting_outputs(spec, recall_payload)
    else:
        setting_root = RUNTIME_SETTINGS_ROOT / str(spec["setting_id"])
        summary_path = setting_root / "setting_summary.json"
        aggregated_path = setting_root / "aggregated_results.json"
        payload = {
            "check_only": True,
            "setting": spec,
            "setting_dir": str(setting_root),
            "runner_cmd": _build_runner_cmd(args, spec, reuse_remote_workspace=reuse_remote_workspace),
            "cached_summary": summary_path.exists(),
            "cached_aggregated_results": aggregated_path.exists(),
        }
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "recall_at_10" in summary:
                payload["cached_recall_at_10"] = float(summary["recall_at_10"])
            payload["summary_path"] = str(summary_path)
        else:
            payload["note"] = "No cached summary found; check-only skips local recall/index rebuild."
        print(json.dumps(payload, indent=2), flush=True)


def main() -> int:
    args = parse_args()
    data_root = Path(str(args.data_root)).expanduser().resolve() if str(args.data_root).strip() else DATA_ROOT.resolve()
    stress_data_root = (
        Path(str(args.stress_data_root)).expanduser().resolve()
        if str(args.stress_data_root).strip()
        else STRESS_DATA_ROOT.resolve()
    )
    _set_runtime_paths(data_root=data_root, stress_data_root=stress_data_root)
    os.environ["VECIO_FAISS_THREADS"] = str(max(1, int(args.faiss_threads)))
    os.environ["VECIO_FAISS_THREADS_DEFAULT"] = str(max(1, int(args.faiss_threads)))
    specs = _selected_specs(args.settings)
    _write_manifest(args, list(SETTING_SPECS))
    # Default to local execution. When --ssh-target is empty the internal
    # stress runner uses third_party/lance on this host.
    pending_specs = list(specs)
    if not args.check_only and not args.force:
        pending_specs = [spec for spec in specs if not _is_setting_complete(spec, repeats=int(args.repeats))]
    skipped = len(specs) - len(pending_specs)
    if skipped:
        print(f"[resume] skipping {skipped} completed settings", flush=True)
    if not pending_specs:
        print("[resume] all selected settings already complete", flush=True)
        if not args.skip_render and not args.check_only:
            for script in ["generate_table.py", "generate_matrix.py"]:
                cmd = [
                    sys.executable,
                    str(HERE / script),
                    "--data-root",
                    str(RUNTIME_DATA_ROOT),
                    "--out-dir",
                    str(RUNTIME_DATA_ROOT),
                ]
                _run(cmd, cwd=ROOT, check_only=False)
        return 0
    reuse_remote_workspace = bool(args.remote_export) and not bool(args.force)
    for idx, spec in enumerate(pending_specs):
        print(f"[setting] {spec['setting_id']} :: {spec['setting_label']}", flush=True)
        _run_one_setting(args, spec, reuse_remote_workspace=bool(idx > 0) or reuse_remote_workspace)
    if not args.skip_render and not args.check_only:
        for script in ["generate_table.py", "generate_matrix.py"]:
            cmd = [
                sys.executable,
                str(HERE / script),
                "--data-root",
                str(RUNTIME_DATA_ROOT),
                "--out-dir",
                str(RUNTIME_DATA_ROOT),
            ]
            _run(cmd, cwd=ROOT, check_only=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
prepare_dataset.py

Prepare and convert datasets into a unified, mmap-friendly on-disk format:

  data/processed/<dataset_key>/
    meta.json
    base.npy
    queries.npy
    gt_I.npy (optional)
    gt_D.npy (optional)

This lets the pipeline load large datasets via np.load(..., mmap_mode='r')
without forcing full RAM materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

WORKING_BASE_ROWS = 1_000_000
DEFAULT_SLICE_SEED = 12345
DATASET_OUTPUT_CONTRACTS = {
    "sift1m": {
        "metric": "euclidean",
        "base": {"rows": [1_000_000], "dim": 128, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 128, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": {"rows": [10_000], "dim": 100, "dtype": "int32", "path": "gt_I.npy"},
    },
    "sift10m": {
        "metric": "euclidean",
        "base": {"rows": [10_000_000], "dim": 128, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 128, "dtype": "float32", "path": "queries.npy"},
    },
    "sift100m": {
        "metric": "euclidean",
        "base": {"rows": [100_000_000], "dim": 128, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 128, "dtype": "float32", "path": "queries.npy"},
    },
    "gist1m": {
        "metric": "euclidean",
        "base": {"rows": [1_000_000], "dim": 960, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 960, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": {"rows": [10_000], "dim": 100, "dtype": "int64", "path": "gt_I.npy"},
    },
    "deep1m": {
        "metric": "angular",
        "normalized": True,
        "base": {"rows": [1_000_000], "dim": 96, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 96, "dtype": "float32", "path": "queries.npy"},
    },
    "deep1b": {
        "metric": "euclidean",
        "base": {"rows": [1_000_000], "dim": 96, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000, 100_000], "dim": 96, "dtype": "float32", "path": "queries.npy"},
    },
    "deep10m": {
        "metric": "euclidean",
        "base": {"rows": [1_000_000], "dim": 96, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000, 100_000], "dim": 96, "dtype": "float32", "path": "queries.npy"},
    },
    "deep100m": {
        "metric": "euclidean",
        "base": {"rows": [100_000_000], "dim": 96, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000, 100_000], "dim": 96, "dtype": "float32", "path": "queries.npy"},
    },
    "msong": {
        "metric": "angular",
        "normalized": True,
        "base": {"rows": [992_272], "dim": 420, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 420, "dtype": "float32", "path": "queries.npy"},
    },
    "openimage": {
        "metric": "euclidean",
        "base": {"rows": [1_000_000], "dim": 512, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 512, "dtype": "float32", "path": "queries.npy"},
    },
    "glove": {
        "metric": "angular",
        "normalized": True,
        "base": {"rows": [1_183_514], "dim": 100, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 100, "dtype": "float32", "path": "queries.npy"},
    },
    "random1m128": {
        "metric": "euclidean",
        "base": {"rows": [1_000_000], "dim": 128, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": [10_000], "dim": 128, "dtype": "float32", "path": "queries.npy"},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_meta(out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=False))


def _open_out_npy(path: Path, shape: Tuple[int, int], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    # open_memmap creates a .npy header + a data section that is memmap-backed.
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def _infer_fvecs_shape(path: Path) -> Tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0].item()
    rec_bytes = 4 * (1 + d)
    if size % rec_bytes != 0:
        raise ValueError(f"Invalid fvecs file size: {path} (size {size} not multiple of {rec_bytes})")
    n = size // rec_bytes
    return int(n), int(d)


def _infer_ivecs_shape(path: Path) -> Tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0].item()
    rec_bytes = 4 * (1 + d)
    if size % rec_bytes != 0:
        raise ValueError(f"Invalid ivecs file size: {path} (size {size} not multiple of {rec_bytes})")
    n = size // rec_bytes
    return int(n), int(d)


def _infer_bvecs_shape(path: Path) -> Tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0].item()
    rec_bytes = 4 + d
    if size % rec_bytes != 0:
        raise ValueError(f"Invalid bvecs file size: {path} (size {size} not multiple of {rec_bytes})")
    n = size // rec_bytes
    return int(n), int(d)


def _read_u8bin_header(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        hdr = np.fromfile(f, dtype=np.uint32, count=2)
    if len(hdr) != 2:
        raise ValueError(f"Invalid u8bin header: {path}")
    return int(hdr[0]), int(hdr[1])


def convert_fvecs_to_npy(in_path: Path, out_path: Path, chunk_rows: int = 65536) -> Tuple[int, int]:
    n, d = _infer_fvecs_shape(in_path)
    out = _open_out_npy(out_path, (n, d), np.float32)

    rec_bytes = 4 * (1 + d)
    with in_path.open("rb") as f:
        written = 0
        while written < n:
            rows = min(chunk_rows, n - written)
            buf = f.read(rows * rec_bytes)
            if len(buf) != rows * rec_bytes:
                raise EOFError(f"Unexpected EOF while reading {in_path}")
            # Interpret as int32 records (d + payload), then reinterpret payload as float32.
            arr_i32 = np.frombuffer(buf, dtype=np.int32).reshape(rows, d + 1)
            dims = arr_i32[:, 0]
            if not np.all(dims == d):
                raise ValueError(f"Bad dimension header in {in_path} near row {written}")
            out[written:written + rows] = arr_i32[:, 1:].view(np.float32)
            written += rows

    out.flush()
    return n, d


def convert_bvecs_to_npy(in_path: Path, out_path: Path, chunk_rows: int = 65536,
                         max_rows: Optional[int] = None) -> Tuple[int, int]:
    n_all, d = _infer_bvecs_shape(in_path)
    n = n_all if max_rows is None else min(int(max_rows), n_all)
    out = _open_out_npy(out_path, (n, d), np.float32)

    rec_bytes = 4 + d
    with in_path.open("rb") as f:
        written = 0
        while written < n:
            rows = min(chunk_rows, n - written)
            buf = f.read(rows * rec_bytes)
            if len(buf) != rows * rec_bytes:
                raise EOFError(f"Unexpected EOF while reading {in_path}")

            dims = np.ndarray(shape=(rows,), dtype=np.int32, buffer=buf, offset=0, strides=(rec_bytes,))
            if not np.all(dims == d):
                raise ValueError(f"Bad dimension header in {in_path} near row {written}")

            vectors = np.ndarray(shape=(rows, d), dtype=np.uint8, buffer=buf, offset=4, strides=(rec_bytes, 1))
            out[written:written + rows] = vectors.astype(np.float32)
            written += rows

    out.flush()
    return n, d


def convert_u8bin_to_npy(in_path: Path, out_path: Path, chunk_rows: int = 65536,
                         max_rows: Optional[int] = None) -> Tuple[int, int]:
    n_all, d = _read_u8bin_header(in_path)
    n = n_all if max_rows is None else min(int(max_rows), n_all)
    expected_min_size = 8 + n * d
    size = in_path.stat().st_size
    if size < expected_min_size:
        raise ValueError(
            f"u8bin file too small for requested rows: {in_path} "
            f"(size={size}, need at least {expected_min_size})"
        )

    out = _open_out_npy(out_path, (n, d), np.float32)
    inp = np.memmap(str(in_path), dtype=np.uint8, mode="r", offset=8, shape=(n, d))
    written = 0
    while written < n:
        rows = min(chunk_rows, n - written)
        out[written:written + rows] = inp[written:written + rows].astype(np.float32)
        written += rows

    out.flush()
    return n, d


def convert_ivecs_to_npy(in_path: Path, out_path: Path, chunk_rows: int = 65536) -> Tuple[int, int]:
    n, d = _infer_ivecs_shape(in_path)
    out = _open_out_npy(out_path, (n, d), np.int32)

    rec_bytes = 4 * (1 + d)
    with in_path.open("rb") as f:
        written = 0
        while written < n:
            rows = min(chunk_rows, n - written)
            buf = f.read(rows * rec_bytes)
            if len(buf) != rows * rec_bytes:
                raise EOFError(f"Unexpected EOF while reading {in_path}")
            arr_i32 = np.frombuffer(buf, dtype=np.int32).reshape(rows, d + 1)
            dims = arr_i32[:, 0]
            if not np.all(dims == d):
                raise ValueError(f"Bad dimension header in {in_path} near row {written}")
            out[written:written + rows] = arr_i32[:, 1:]
            written += rows

    out.flush()
    return n, d


def _read_bin_header(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        nb_d = np.fromfile(f, dtype=np.int32, count=2)
        if len(nb_d) != 2:
            raise ValueError(f"Invalid bin header: {path}")
        nb, d = int(nb_d[0]), int(nb_d[1])
    return nb, d


def _sample_row_indices(total_rows: int, target_rows: int, seed: int) -> np.ndarray:
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    target_rows = min(int(target_rows), int(total_rows))
    if target_rows >= int(total_rows):
        return np.arange(int(total_rows), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(total_rows), size=int(target_rows), replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)


def _copy_selected_rows_to_npy(
    src,
    out_path: Path,
    row_indices: np.ndarray,
    *,
    chunk_rows: int,
) -> Tuple[int, int]:
    idx = np.asarray(row_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("row_indices must be 1D")
    if idx.size == 0:
        raise ValueError("row_indices must not be empty")
    if np.any(idx[1:] < idx[:-1]):
        raise ValueError("row_indices must be sorted")

    dim = int(src.shape[1])
    out = _open_out_npy(out_path, (int(idx.size), dim), np.float32)
    written = 0
    while written < len(idx):
        rows = idx[written:written + chunk_rows]
        out[written:written + len(rows)] = np.asarray(src[rows], dtype=np.float32)
        written += len(rows)
    out.flush()
    return int(idx.size), dim


def convert_bin_to_npy(in_path: Path, out_path: Path, chunk_rows: int = 65536) -> Tuple[int, int]:
    nb, d = _read_bin_header(in_path)
    out = _open_out_npy(out_path, (nb, d), np.float32)

    # Data is contiguous float32 after 8-byte header.
    inp = np.memmap(str(in_path), dtype=np.float32, mode="r", offset=8, shape=(nb, d))
    written = 0
    while written < nb:
        rows = min(chunk_rows, nb - written)
        out[written:written + rows] = inp[written:written + rows]
        written += rows

    out.flush()
    return nb, d


def convert_bin_to_npy_subset(
    in_path: Path,
    out_path: Path,
    *,
    row_indices: np.ndarray,
    chunk_rows: int = 65536,
) -> Tuple[int, int]:
    nb, d = _read_bin_header(in_path)
    idx = np.asarray(row_indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("row_indices must not be empty")
    if int(idx[0]) < 0 or int(idx[-1]) >= nb:
        raise ValueError(f"row_indices out of bounds for {in_path}: nb={nb}")
    inp = np.memmap(str(in_path), dtype=np.float32, mode="r", offset=8, shape=(nb, d))
    return _copy_selected_rows_to_npy(inp, out_path, idx, chunk_rows=chunk_rows)


def _default_processed_dir(dataset_key: str) -> Path:
    return Path("data") / "processed" / dataset_key


def _download_file(url: str, dst: Path, expected_bytes: Optional[int] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    # Some hosting providers block non-browser UAs or require a Referer.
    # Use conservative browser-like headers and follow redirects via urllib.
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        # Often required by bot filters when requesting a "download" URL directly.
        "Referer": "https://ann-benchmarks.com/",
        "Connection": "close",
    }

    def _download_via_urllib() -> None:
        req = urllib.request.Request(url, headers=headers)
        # Keep a finite timeout so CI/sandbox won't hang forever on bad networking.
        with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
            total = r.length if getattr(r, "length", None) is not None else expected_bytes
            done = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and total > 0:
                    pct = 100.0 * done / total
                    print(f"\rDownloading {dst.name}: {pct:5.1f}% ({done}/{total} bytes)", end="")
                else:
                    print(f"\rDownloading {dst.name}: {done} bytes", end="")
        print("")

    def _download_via_curl() -> None:
        curl = shutil.which("curl")
        if not curl:
            raise FileNotFoundError("curl not found")
        cmd = [
            curl,
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "1",
            "--retry-connrefused",
            "-A",
            headers["User-Agent"],
            "-e",
            headers["Referer"],
            "-o",
            str(tmp),
            url,
        ]
        # Let curl show progress; it's usually more robust against odd CDN behavior.
        subprocess.run(cmd, check=True)

    try:
        _download_via_urllib()
    except Exception as e:
        # Some CDNs return 403 to urllib while allowing curl (different TLS/HTTP stack).
        try:
            _download_via_curl()
        except Exception as e2:
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(
                f"Failed to download {url}. If you're running in an offline/sandboxed environment, "
                f"download it manually and re-run without --download. Error: {e} ; curl fallback: {e2}"
            ) from e2

    tmp.rename(dst)
    if expected_bytes is not None:
        got = dst.stat().st_size
        if got != expected_bytes:
            raise ValueError(f"Downloaded size mismatch for {dst}: got {got}, expected {expected_bytes}")


def _ensure_sift1m_raw(download: bool) -> None:
    raw_dir = Path("data") / "sift1m"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_f = raw_dir / "sift_base.fvecs"
    query_f = raw_dir / "sift_query.fvecs"
    gt_f = raw_dir / "sift_groundtruth.ivecs"
    if base_f.exists() and query_f.exists() and gt_f.exists():
        return

    tar_path = Path("data") / "sift.tar.gz"
    if not tar_path.exists():
        if not download:
            raise FileNotFoundError(
                "SIFT1M raw files not found. Run with --download or place raw files under data/sift1m/."
            )
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        expected = 168280445
        _download_file(url, tar_path, expected_bytes=expected)

    # Extract tarball. It contains a top-level "sift/" directory.
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=raw_dir)

    extracted_dir = raw_dir / "sift"
    if extracted_dir.exists():
        for fname in ("sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs", "sift_learn.fvecs"):
            src = extracted_dir / fname
            if src.exists():
                dst = raw_dir / fname
                if not dst.exists():
                    src.rename(dst)
        # Best-effort cleanup if empty
        try:
            extracted_dir.rmdir()
        except OSError:
            pass

    if not (base_f.exists() and query_f.exists() and gt_f.exists()):
        raise FileNotFoundError(
            "SIFT1M extraction completed but expected files are still missing under data/sift1m/."
        )


def _ensure_gdrive_folder(folder_id: str, out_dir: Path, download: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        # Assume already present; caller will validate exact filenames.
        return
    if not download:
        raise FileNotFoundError(f"Raw dir empty: {out_dir} (run with --download)")

    try:
        import gdown  # type: ignore
    except Exception as e:
        # Fallback to the repo-local venv if available. This lets callers use
        # system python while still reusing a project-scoped gdown install.
        venv_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
        if not venv_python.exists():
            raise RuntimeError(
                "gdown is required for Google Drive downloads but is not available. "
                f"Error: {e}"
            ) from e
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        subprocess.run(
            [
                str(venv_python),
                "-c",
                (
                    "import gdown; "
                    f"gdown.download_folder(url={folder_url!r}, output={str(out_dir)!r}, "
                    "quiet=False, use_cookies=False)"
                ),
            ],
            check=True,
        )
        entries = [p for p in out_dir.iterdir() if p.is_dir()]
        if len(entries) == 1 and not any(p.is_file() for p in out_dir.iterdir()):
            nested = entries[0]
            for child in nested.iterdir():
                child.rename(out_dir / child.name)
            try:
                nested.rmdir()
            except OSError:
                pass
        return

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    # gdown will create a subfolder under out_dir depending on the Drive structure.
    gdown.download_folder(url=url, output=str(out_dir), quiet=False, use_cookies=False)

    # If it created a single nested directory, flatten it.
    entries = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(entries) == 1 and not any(p.is_file() for p in out_dir.iterdir()):
        nested = entries[0]
        for child in nested.iterdir():
            child.rename(out_dir / child.name)
        try:
            nested.rmdir()
        except OSError:
            pass


def _ensure_http_file(url: str, dst: Path, download: bool, expected_bytes: Optional[int] = None) -> None:
    if dst.exists():
        return
    if not download:
        raise FileNotFoundError(f"Missing required file: {dst} (run with --download)")
    _download_file(url, dst, expected_bytes=expected_bytes)


def _ensure_http_file_multi(urls: list[str], dst: Path, download: bool,
                            expected_bytes: Optional[int] = None) -> None:
    if dst.exists():
        return
    if not download:
        raise FileNotFoundError(f"Missing required file: {dst} (run with --download)")
    errors: list[tuple[str, str]] = []
    for url in urls:
        try:
            _download_file(url, dst, expected_bytes=expected_bytes)
            return
        except Exception as e:
            errors.append((url, str(e)))

    msg = [f"All download URLs failed for {dst}:"]
    for url, err in errors:
        msg.append(f"  - {url}: {err}")
    raise RuntimeError("\n".join(msg))


def _download_http_range(url: str, dst: Path, start: int, end: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for ranged dataset downloads")
    cmd = [
        curl,
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "1",
        "--retry-connrefused",
        "-o",
        str(tmp),
        "-r",
        f"{start}-{end}",
        url,
    ]
    subprocess.run(cmd, check=True)
    expected_bytes = end - start + 1
    observed_bytes = tmp.stat().st_size
    if observed_bytes != expected_bytes:
        tmp.unlink()
        raise ValueError(
            f"ranged download returned {observed_bytes} bytes, expected {expected_bytes}; "
            "the server may have ignored the Range header"
        )
    tmp.rename(dst)


def _validate_floatbin_identity(path: Path, *, rows: int, dimension: int) -> None:
    expected_bytes = 8 + rows * dimension * np.dtype(np.float32).itemsize
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"floatbin size mismatch: {path} has {observed_bytes} bytes, "
            f"expected {expected_bytes}"
        )
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int32, count=2)
    if header.tolist() != [rows, dimension]:
        raise ValueError(
            f"floatbin header mismatch: {path} has {header.tolist()}, "
            f"expected {[rows, dimension]}"
        )


def _assemble_floatbin_prefix(
    payload: Path,
    output: Path,
    *,
    rows: int,
    dimension: int,
) -> None:
    """Atomically publish a headerless float32 payload as a floatbin file."""
    expected_payload_bytes = rows * dimension * np.dtype(np.float32).itemsize
    observed_payload_bytes = payload.stat().st_size
    if observed_payload_bytes != expected_payload_bytes:
        raise ValueError(
            f"floatbin payload size mismatch: {payload} has {observed_payload_bytes} "
            f"bytes, expected {expected_payload_bytes}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.asarray([rows, dimension], dtype=np.int32).tofile(destination)
            with payload.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        _validate_floatbin_identity(temporary, rows=rows, dimension=dimension)
        temporary.replace(output)
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _h5_find_keys(f) -> Tuple[str, str]:
    # Ann-benchmarks HDF5 convention: train/test. Try a few fallbacks for robustness.
    keys = set(f.keys())
    for train_k in ("train", "data", "base"):
        if train_k in keys:
            break
    else:
        raise KeyError(f"Could not find base vectors in HDF5; keys={sorted(keys)}")

    for test_k in ("test", "queries", "query"):
        if test_k in keys:
            break
    else:
        raise KeyError(f"Could not find queries in HDF5; keys={sorted(keys)}")

    return train_k, test_k


def convert_hdf5_to_npy(
    h5_path: Path,
    out_base: Path,
    out_queries: Path,
    normalize: bool,
    chunk_rows: int,
    *,
    train_row_indices: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError(f"h5py is required to convert HDF5 datasets: {e}") from e

    with h5py.File(h5_path, "r") as f:
        train_k, test_k = _h5_find_keys(f)
        train = f[train_k]
        test = f[test_k]

        if train.ndim != 2 or test.ndim != 2:
            raise ValueError(f"Expected 2D arrays in HDF5; got train={train.shape}, test={test.shape}")
        nb_full, d0 = int(train.shape[0]), int(train.shape[1])
        nq, d1 = int(test.shape[0]), int(test.shape[1])
        if d0 != d1:
            raise ValueError(f"HDF5 dim mismatch: train dim={d0}, test dim={d1}")

        if train_row_indices is None:
            train_idx = np.arange(nb_full, dtype=np.int64)
        else:
            train_idx = np.asarray(train_row_indices, dtype=np.int64)
            if train_idx.ndim != 1:
                raise ValueError("train_row_indices must be 1D")
            if train_idx.size == 0:
                raise ValueError("train_row_indices must not be empty")
            if np.any(train_idx[1:] < train_idx[:-1]):
                raise ValueError("train_row_indices must be sorted")
            if int(train_idx[0]) < 0 or int(train_idx[-1]) >= nb_full:
                raise ValueError(f"train_row_indices out of bounds for {h5_path}: nb={nb_full}")

        nb = int(train_idx.size)
        out_b = _open_out_npy(out_base, (nb, d0), np.float32)
        out_q = _open_out_npy(out_queries, (nq, d1), np.float32)

        def write_chunks(src, dst, rows_total: int, *, row_indices: Optional[np.ndarray] = None) -> None:
            written = 0
            if row_indices is None:
                while written < rows_total:
                    rows = min(chunk_rows, rows_total - written)
                    arr = np.asarray(src[written:written + rows], dtype=np.float32)
                    if normalize:
                        norms = np.linalg.norm(arr, axis=1, keepdims=True)
                        norms[norms == 0.0] = 1.0
                        arr = arr / norms
                    dst[written:written + rows] = arr
                    written += rows
                return

            while written < len(row_indices):
                idx_chunk = row_indices[written:written + chunk_rows]
                arr = np.asarray(src[idx_chunk], dtype=np.float32)
                if normalize:
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms[norms == 0.0] = 1.0
                    arr = arr / norms
                dst[written:written + len(idx_chunk)] = arr
                written += len(idx_chunk)

        write_chunks(train, out_b, nb, row_indices=train_idx)
        write_chunks(test, out_q, nq)

        out_b.flush()
        out_q.flush()

    return (nb, d0), (nq, d1)


def _prepare_ann_hdf5_dataset(
    dataset_key: str,
    raw_dir: Path,
    out_dir: Path,
    *,
    h5_filename: str,
    urls: list[str],
    chunk_rows: int,
    download: bool,
    target_rows: int,
    slice_seed: int,
    metric: str,
    display_name: str,
    normalize: bool = True,
    sample_mode: str = "random",
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    h5_path = raw_dir / h5_filename
    _ensure_http_file_multi(urls, h5_path, download=download)

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError(f"h5py is required to inspect {display_name} HDF5 dataset: {e}") from e

    with h5py.File(h5_path, "r") as f:
        train_k, _ = _h5_find_keys(f)
        total_base_rows = int(f[train_k].shape[0])

    if sample_mode == "first":
        take = min(int(target_rows), int(total_base_rows))
        base_idx = np.arange(take, dtype=np.int64)
    else:
        base_idx = _sample_row_indices(total_base_rows, int(target_rows), int(slice_seed))
    (nb, d0), (nq, d1) = convert_hdf5_to_npy(
        h5_path,
        out_dir / "base.npy",
        out_dir / "queries.npy",
        normalize=normalize,
        chunk_rows=chunk_rows,
        train_row_indices=base_idx,
    )

    meta = {
        "dataset": dataset_key,
        "format": "clivf-npy-v1",
        "metric": metric,
        "normalization": {"l2_unit": bool(normalize)},
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {
            "raw_file": str(h5_path),
            "urls": urls,
            "full_base_rows": total_base_rows,
        },
        "base_sample": {
            "mode": "prefix" if sample_mode == "first" else "random",
            "seed": None if sample_mode == "first" else int(slice_seed),
            "rows": nb,
        },
    }
    _write_meta(out_dir, meta)


def _prepare_random_dataset(
    dataset_key: str,
    out_dir: Path,
    *,
    nb: int,
    nq: int,
    dim: int,
    seed: int,
    chunk_rows: int,
) -> None:
    rng = np.random.default_rng(int(seed))
    out_b = _open_out_npy(out_dir / "base.npy", (int(nb), int(dim)), np.float32)
    out_q = _open_out_npy(out_dir / "queries.npy", (int(nq), int(dim)), np.float32)

    written = 0
    while written < nb:
        rows = min(chunk_rows, nb - written)
        out_b[written:written + rows] = rng.standard_normal((rows, dim), dtype=np.float32)
        written += rows

    written = 0
    while written < nq:
        rows = min(chunk_rows, nq - written)
        out_q[written:written + rows] = rng.standard_normal((rows, dim), dtype=np.float32)
        written += rows

    out_b.flush()
    out_q.flush()

    meta = {
        "dataset": dataset_key,
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": int(nb), "dim": int(dim), "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": int(nq), "dim": int(dim), "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {"generator": "numpy.standard_normal", "seed": int(seed)},
    }
    _write_meta(out_dir, meta)


def prepare_sift1m(out_dir: Path, chunk_rows: int, download: bool) -> None:
    raw_dir = Path("data") / "sift1m"
    base_f = raw_dir / "sift_base.fvecs"
    query_f = raw_dir / "sift_query.fvecs"
    gt_f = raw_dir / "sift_groundtruth.ivecs"

    _ensure_sift1m_raw(download=download)

    nb, d0 = convert_fvecs_to_npy(base_f, out_dir / "base.npy", chunk_rows=chunk_rows)
    nq, d1 = convert_fvecs_to_npy(query_f, out_dir / "queries.npy", chunk_rows=chunk_rows)
    if d0 != d1:
        raise ValueError(f"SIFT1M base/query dim mismatch: {d0} vs {d1}")
    gt_nq, gt_k = convert_ivecs_to_npy(gt_f, out_dir / "gt_I.npy", chunk_rows=chunk_rows)
    if gt_nq != nq:
        # Some datasets have gt for all 10k queries even if you later subset; that's OK.
        pass

    meta = {
        "dataset": "sift1m",
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": {"rows": gt_nq, "k": gt_k, "dtype": "int32", "I_path": "gt_I.npy", "D_path": None},
        "source": {"raw_dir": str(raw_dir)},
    }
    _write_meta(out_dir, meta)


def prepare_sift10m(out_dir: Path, chunk_rows: int, download: bool) -> None:
    raw_dir = Path("data") / "sift10m"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_u8bin = raw_dir / "learn.10M.u8bin"
    query_u8bin = raw_dir / "query.public.10K.u8bin"
    legacy_raw_dir = Path("data") / "sift100m"
    base_f = legacy_raw_dir / "bigann_base.bvecs"
    query_f = legacy_raw_dir / "bigann_query.bvecs"
    subset_rows = 10_000_000
    d = 128

    if not base_u8bin.exists() or not query_u8bin.exists():
        if download:
            base_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/learn.100M.u8bin"
            query_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"
            if not base_u8bin.exists():
                data_tmp = raw_dir / "learn.10M.data"
                byte_count = subset_rows * d
                _download_http_range(base_url, data_tmp, 8, 8 + byte_count - 1)
                with base_u8bin.open("wb") as f:
                    np.array([subset_rows, d], dtype=np.uint32).tofile(f)
                    with data_tmp.open("rb") as fin:
                        shutil.copyfileobj(fin, f, length=1024 * 1024 * 16)
                data_tmp.unlink()
            if not query_u8bin.exists():
                _ensure_http_file(query_url, query_u8bin, download=True)
        elif not (base_f.exists() and query_f.exists()):
            raise FileNotFoundError(
                "SIFT10M preparation requires either raw files under data/sift10m/ "
                "(learn.10M.u8bin, query.public.10K.u8bin) or legacy BigANN files under "
                "data/sift100m/ (bigann_base.bvecs, bigann_query.bvecs)."
            )

    if base_u8bin.exists() and query_u8bin.exists():
        nb, d0 = convert_u8bin_to_npy(base_u8bin, out_dir / "base.npy", chunk_rows=chunk_rows)
        nq, d1 = convert_u8bin_to_npy(query_u8bin, out_dir / "queries.npy", chunk_rows=chunk_rows)
        source = {
            "raw_dir": str(raw_dir),
            "base_file": base_u8bin.name,
            "query_file": query_u8bin.name,
            "base_prefix_rows": subset_rows,
            "download_source": "big-ann-benchmarks official u8bin",
        }
    else:
        nb, d0 = convert_bvecs_to_npy(
            base_f,
            out_dir / "base.npy",
            chunk_rows=chunk_rows,
            max_rows=subset_rows,
        )
        nq, d1 = convert_bvecs_to_npy(query_f, out_dir / "queries.npy", chunk_rows=chunk_rows)
        source = {
            "raw_dir": str(legacy_raw_dir),
            "base_prefix_rows": subset_rows,
            "note": "Uses the first 10M vectors from legacy BigANN bvecs.",
        }
    if d0 != d1:
        raise ValueError(f"SIFT10M base/query dim mismatch: {d0} vs {d1}")

    meta = {
        "dataset": "sift10m",
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": source,
    }
    _write_meta(out_dir, meta)


def prepare_sift100m(out_dir: Path, chunk_rows: int, download: bool) -> None:
    """SIFT100M: full BigANN learn.100M.u8bin (100M x 128 uint8).

    Unlike sift10m which takes a prefix, sift100m uses the entire source
    file. The public u8bin format is::

        <uint32 n><uint32 d><uint8[n*d] data>

    so no byte-range trickery is needed — we just download the full
    file (~12.8 GB + header) and convert.
    """
    raw_dir = Path("data") / "sift100m_u8bin"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_u8bin = raw_dir / "learn.100M.u8bin"
    query_u8bin = raw_dir / "query.public.10K.u8bin"
    subset_rows = 100_000_000
    d = 128

    if not base_u8bin.exists() or not query_u8bin.exists():
        if not download:
            raise FileNotFoundError(
                "SIFT100M requires --download or existing raw files under "
                f"{raw_dir}/."
            )
        base_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/learn.100M.u8bin"
        query_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"
        if not base_u8bin.exists():
            _ensure_http_file(base_url, base_u8bin, download=True)
        if not query_u8bin.exists():
            _ensure_http_file(query_url, query_u8bin, download=True)

    nb, d0 = convert_u8bin_to_npy(base_u8bin, out_dir / "base.npy", chunk_rows=chunk_rows)
    nq, d1 = convert_u8bin_to_npy(query_u8bin, out_dir / "queries.npy", chunk_rows=chunk_rows)
    if d0 != d1:
        raise ValueError(f"SIFT100M base/query dim mismatch: {d0} vs {d1}")
    if nb != subset_rows:
        raise ValueError(f"SIFT100M base rows {nb} != expected {subset_rows}")

    meta = {
        "dataset": "sift100m",
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {
            "raw_dir": str(raw_dir),
            "base_file": base_u8bin.name,
            "query_file": query_u8bin.name,
            "base_prefix_rows": subset_rows,
            "download_source": "big-ann-benchmarks official u8bin (full)",
        },
    }
    _write_meta(out_dir, meta)


def prepare_coco(out_dir: Path, chunk_rows: int, download: bool) -> None:
    raw_dir = Path("data") / "COCO"
    base_f = raw_dir / "data_100000_768"
    query_f = raw_dir / "queries_500_768"

    _ensure_gdrive_folder("1Hp6SI8YOFPdWbmC1a4_-1dZWxZH3CHMS", raw_dir, download=download)

    if not (base_f.exists() and query_f.exists()):
        raise FileNotFoundError(
            "COCO raw files not found under data/COCO/. "
            "Expected data_100000_768 and queries_500_768."
        )

    nb, d0 = convert_bin_to_npy(base_f, out_dir / "base.npy", chunk_rows=chunk_rows)
    nq, d1 = convert_bin_to_npy(query_f, out_dir / "queries.npy", chunk_rows=chunk_rows)
    if d0 != d1:
        raise ValueError(f"COCO base/query dim mismatch: {d0} vs {d1}")

    meta = {
        "dataset": "coco",
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {"raw_dir": str(raw_dir)},
    }
    _write_meta(out_dir, meta)


def prepare_openimage(out_dir: Path, chunk_rows: int, download: bool) -> None:
    raw_dir = Path("data") / "openimage-streaming"
    base_f = raw_dir / "data_1000000_512"
    query_f = raw_dir / "queries_10000_512"

    # The default Google Drive folder ID below was supplied by the
    # paper authors. Reviewers can override it with
    # VECIO_OPENIMAGE_FOLDER_ID if the original folder ever expires
    # or rate-limits gdown. The expected raw files (data_1000000_512
    # and queries_10000_512) must be at the top level of the folder.
    folder_id = os.environ.get(
        "VECIO_OPENIMAGE_FOLDER_ID",
        "1ZkWOrja-0A6C9yh3ysFoCP6w5u7oWjQx",
    )
    _ensure_gdrive_folder(folder_id, raw_dir, download=download)

    if not (base_f.exists() and query_f.exists()):
        raise FileNotFoundError(
            "OpenImage raw files not found under data/openimage-streaming/. "
            "Expected data_1000000_512 and queries_10000_512. "
            "Set VECIO_OPENIMAGE_FOLDER_ID to a Google Drive folder "
            "containing those filenames if the default folder has expired."
        )

    nb, d0 = convert_bin_to_npy(base_f, out_dir / "base.npy", chunk_rows=chunk_rows)
    nq, d1 = convert_bin_to_npy(query_f, out_dir / "queries.npy", chunk_rows=chunk_rows)
    if d0 != d1:
        raise ValueError(f"OpenImage base/query dim mismatch: {d0} vs {d1}")

    meta = {
        "dataset": "openimage",
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {"raw_dir": str(raw_dir)},
    }
    _write_meta(out_dir, meta)


def prepare_glove(
    out_dir: Path,
    chunk_rows: int,
    download: bool,
    *,
    target_rows: int = 1_183_514,
    slice_seed: int = DEFAULT_SLICE_SEED,
) -> None:
    """
    Download and prepare the GloVe 100D ANN benchmark dataset.

    Source: ann-benchmarks HDF5 dataset (angular metric).
    We L2-normalize vectors by default so L2 distance is comparable to cosine.
    """
    raw_dir = Path("data") / "GLOVE"
    raw_dir.mkdir(parents=True, exist_ok=True)

    h5_path = raw_dir / "glove-100-angular.hdf5"
    # Prefer HTTP first: some setups return 403 for HTTPS, others the opposite.
    urls = [
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "https://ann-benchmarks.com/glove-100-angular.hdf5",
    ]
    # Escape hatch: allow overriding URLs without changing code.
    # Example:
    #   VECIO_GLOVE_URLS="https://mirror1/file.hdf5,https://mirror2/file.hdf5" ... --download
    env_urls = os.environ.get("VECIO_GLOVE_URLS") or os.environ.get("VECIO_GLOVE_URL")
    if env_urls:
        urls = [u.strip() for u in env_urls.split(",") if u.strip()]
    _ensure_http_file_multi(urls, h5_path, download=download)

    # Ann-benchmarks is "angular". Normalize vectors so L2 works as cosine surrogate.
    normalize = True
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError(f"h5py is required to inspect GloVe HDF5 datasets: {e}") from e

    with h5py.File(h5_path, "r") as f:
        train_k, _ = _h5_find_keys(f)
        total_base_rows = int(f[train_k].shape[0])

    base_idx = _sample_row_indices(total_base_rows, int(target_rows), int(slice_seed))
    (nb, d0), (nq, d1) = convert_hdf5_to_npy(
        h5_path,
        out_dir / "base.npy",
        out_dir / "queries.npy",
        normalize=normalize,
        chunk_rows=chunk_rows,
        train_row_indices=base_idx,
    )

    meta = {
        "dataset": "glove",
        "format": "clivf-npy-v1",
        "metric": "angular",
        "normalization": {"l2_unit": bool(normalize)},
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {"raw_file": str(h5_path), "urls": urls, "full_base_rows": total_base_rows},
        "base_sample": {
            "mode": "full" if nb == total_base_rows else "random",
            "seed": None if nb == total_base_rows else int(slice_seed),
            "rows": nb,
        },
    }
    _write_meta(out_dir, meta)


def repeated_query_pool(
    queries: np.ndarray,
    *,
    repeats: int,
    seed: int = 20260811,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand queries using the exact historical per-epoch permutation policy."""
    values = np.asarray(queries, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("queries must be a non-empty two-dimensional array")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    epochs = [
        np.random.default_rng([seed, epoch]).permutation(len(values))
        for epoch in range(repeats)
    ]
    origin = np.concatenate(epochs).astype(np.int64, copy=False)
    return np.ascontiguousarray(values[origin], dtype=np.float32), origin


def prepare_gist1m(out_dir: Path, chunk_rows: int, download: bool) -> None:
    raw_dir = Path("data") / "gist1m"
    raw_dir.mkdir(parents=True, exist_ok=True)
    h5_path = raw_dir / "gist-960-euclidean.hdf5"
    source_sha256 = "8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a"
    urls = [
        "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    ]
    _ensure_http_file_multi(urls, h5_path, download=download)
    observed_sha256 = _sha256(h5_path)
    if observed_sha256 != source_sha256:
        raise ValueError(
            f"GIST source SHA-256 mismatch: {observed_sha256} != {source_sha256}"
        )
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"h5py is required to prepare GIST1M: {exc}") from exc
    with h5py.File(h5_path, "r") as handle:
        required = {"train", "test", "neighbors"}
        if not required.issubset(handle.keys()):
            raise ValueError(f"GIST source lacks datasets: {sorted(required - set(handle.keys()))}")
        train = handle["train"]
        standard_queries = np.asarray(handle["test"], dtype=np.float32)
        standard_gt = np.asarray(handle["neighbors"], dtype=np.int64)
        if train.shape != (1_000_000, 960):
            raise ValueError(f"unexpected GIST base shape: {train.shape}")
        if standard_queries.shape != (1_000, 960):
            raise ValueError(f"unexpected GIST query shape: {standard_queries.shape}")
        if standard_gt.shape != (1_000, 100):
            raise ValueError(f"unexpected GIST ground-truth shape: {standard_gt.shape}")
        base_shape = tuple(map(int, train.shape))
        base = _open_out_npy(out_dir / "base.npy", base_shape, np.float32)
        for start in range(0, len(train), chunk_rows):
            stop = min(start + chunk_rows, len(train))
            base[start:stop] = np.asarray(train[start:stop], dtype=np.float32)
        base.flush()
    np.save(out_dir / "queries_standard_1k.npy", standard_queries)
    np.save(out_dir / "gt_I_standard_1k.npy", standard_gt)
    measured_queries, query_origin = repeated_query_pool(standard_queries, repeats=10)
    measured_gt = np.ascontiguousarray(standard_gt[query_origin], dtype=np.int64)
    np.save(out_dir / "query_origin_10k.npy", query_origin)
    np.save(out_dir / "queries.npy", measured_queries)
    np.save(out_dir / "gt_I.npy", measured_gt)
    nb, d0 = base_shape
    nq, d1 = measured_queries.shape
    ground_truth = {"rows": len(measured_gt), "k": 100, "I_path": "gt_I.npy"}

    meta = {
        "dataset": "gist1m",
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": ground_truth,
        "source": {
            "raw_file": str(h5_path),
            "url": urls[0],
            "sha256": observed_sha256,
        },
        "query_expansion": {
            "source_queries": 1_000,
            "uniform_repetitions": 10,
            "permutation_seed": 20260811,
            "origin_path": "query_origin_10k.npy",
            "standard_queries_path": "queries_standard_1k.npy",
            "standard_ground_truth_path": "gt_I_standard_1k.npy",
        },
    }
    _write_meta(out_dir, meta)


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _prepare_deep_floatbin_dataset(
    dataset_key: str,
    raw_dir: Path,
    out_dir: Path,
    *,
    chunk_rows: int,
    download: bool,
    target_rows: int,
    slice_seed: int,
    base_candidates: list[str],
    query_candidates: list[str],
    display_name: str,
    sample_mode: str = "random",
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_f = _first_existing([raw_dir / name for name in base_candidates])
    query_f = _first_existing([raw_dir / name for name in query_candidates])

    if base_f is None or query_f is None:
        if download:
            raise RuntimeError(
                f"Automatic {display_name} download is not implemented. "
                f"Place raw files under {raw_dir}/ first."
            )
        raise FileNotFoundError(
            f"{display_name} raw files not found under {raw_dir}/. "
            f"Expected one of {base_candidates} with one of {query_candidates}."
        )

    total_base_rows, _ = _read_bin_header(base_f)
    if sample_mode == "first":
        base_idx = np.arange(min(total_base_rows, int(target_rows)), dtype=np.int64)
    else:
        base_idx = _sample_row_indices(total_base_rows, int(target_rows), int(slice_seed))
    nb, d0 = convert_bin_to_npy_subset(
        base_f,
        out_dir / "base.npy",
        row_indices=base_idx,
        chunk_rows=chunk_rows,
    )
    nq, d1 = convert_bin_to_npy(query_f, out_dir / "queries.npy", chunk_rows=chunk_rows)
    if d0 != d1:
        raise ValueError(f"{display_name} base/query dim mismatch: {d0} vs {d1}")

    meta = {
        "dataset": dataset_key,
        "format": "clivf-npy-v1",
        "metric": "euclidean",
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d1, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {
            "raw_dir": str(raw_dir),
            "base_file": str(base_f.name),
            "query_file": str(query_f.name),
            "full_base_rows": int(total_base_rows),
        },
        "base_sample": {
            "mode": "prefix" if sample_mode == "first" else "random",
            "seed": None if sample_mode == "first" else int(slice_seed),
            "rows": nb,
        },
    }
    _write_meta(out_dir, meta)


def prepare_deep1b(
    out_dir: Path,
    chunk_rows: int,
    download: bool,
    *,
    target_rows: int = WORKING_BASE_ROWS,
    slice_seed: int = DEFAULT_SLICE_SEED,
) -> None:
    raw_dir = Path("data") / "deep1b"
    _prepare_deep_floatbin_dataset(
        "deep1b",
        raw_dir,
        out_dir,
        chunk_rows=chunk_rows,
        download=download,
        target_rows=target_rows,
        slice_seed=slice_seed,
        base_candidates=["base.1B.fbin", "base.fbin"],
        query_candidates=["query.public.10K.fbin", "query.public.100K.fbin", "query.fbin"],
        display_name="DEEP1B",
    )


def prepare_deep1m(
    out_dir: Path,
    chunk_rows: int,
    download: bool,
    *,
    target_rows: int = WORKING_BASE_ROWS,
    slice_seed: int = DEFAULT_SLICE_SEED,
) -> None:
    raw_dir = Path("data") / "deep1m"
    urls = [
        "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        "https://ann-benchmarks.com/deep-image-96-angular.hdf5",
    ]
    _prepare_ann_hdf5_dataset(
        "deep1m",
        raw_dir,
        out_dir,
        h5_filename="deep-image-96-angular.hdf5",
        urls=urls,
        chunk_rows=chunk_rows,
        download=download,
        target_rows=target_rows,
        slice_seed=slice_seed,
        metric="angular",
        display_name="DEEP1M",
        normalize=True,
        sample_mode="first",
    )


def prepare_deep10m(
    out_dir: Path,
    chunk_rows: int,
    download: bool,
    *,
    target_rows: int = WORKING_BASE_ROWS,
    slice_seed: int = DEFAULT_SLICE_SEED,
) -> None:
    raw_dir = Path("data") / "deep10m"
    _prepare_deep_floatbin_dataset(
        "deep10m",
        raw_dir,
        out_dir,
        chunk_rows=chunk_rows,
        download=download,
        target_rows=target_rows,
        slice_seed=slice_seed,
        base_candidates=["base.1B.fbin.crop_nb_10000000", "base.10M.fbin", "base.fbin"],
        query_candidates=["query.public.100K.fbin", "query.public.10K.fbin", "query.fbin"],
        display_name="DEEP10M",
    )


def prepare_deep100m(
    out_dir: Path,
    chunk_rows: int,
    download: bool,
    *,
    target_rows: int = 100_000_000,
    slice_seed: int = DEFAULT_SLICE_SEED,
) -> None:
    """Prepare the first 100M DEEP1B vectors used by the paper campaign."""
    raw_dir = Path("data") / "deep1b"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = raw_dir / "base.100M.fbin"
    query = raw_dir / "query.public.10K.fbin"
    prefix_valid = False
    if prefix.is_file():
        try:
            _validate_floatbin_identity(prefix, rows=target_rows, dimension=96)
            prefix_valid = True
        except ValueError:
            if not download:
                raise
    if download and not prefix_valid:
        payload = raw_dir / "base.100M.payload"
        bytes_needed = target_rows * 96 * 4
        _download_http_range(
            "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin",
            payload,
            8,
            8 + bytes_needed - 1,
        )
        _assemble_floatbin_prefix(
            payload,
            prefix,
            rows=target_rows,
            dimension=96,
        )
        payload.unlink(missing_ok=True)
    if download and not query.is_file():
        _ensure_http_file(
            "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin",
            query,
            download=True,
        )
    _prepare_deep_floatbin_dataset(
        "deep100m",
        raw_dir,
        out_dir,
        chunk_rows=chunk_rows,
        download=download,
        target_rows=target_rows,
        slice_seed=slice_seed,
        base_candidates=["base.100M.fbin", "base.1B.fbin", "base.fbin"],
        query_candidates=["query.public.10K.fbin", "query.public.100K.fbin", "query.fbin"],
        display_name="DEEP100M",
        sample_mode="first",
    )


def prepare_msong(
    out_dir: Path,
    chunk_rows: int,
    download: bool,
    *,
    target_rows: int = WORKING_BASE_ROWS,
    slice_seed: int = DEFAULT_SLICE_SEED,
) -> None:
    raw_dir = Path("data") / "msong"
    raw_dir.mkdir(parents=True, exist_ok=True)
    h5_path = raw_dir / "msong-420-angular.hdf5"
    raw_bin = raw_dir / "data_992272_420"
    # Default Google Drive folder ID; overridable via env var so that
    # reviewers can point at a private mirror if the original folder
    # ever expires or rate-limits gdown.
    default_folder_id = os.environ.get(
        "VECIO_MSONG_FOLDER_ID",
        "1TnLNJNVqyFrEzKGfQVdvUC8Al-tmjVg0",
    )

    urls: list[str] = []
    env_urls = os.environ.get("VECIO_MSONG_URLS") or os.environ.get("VECIO_MSONG_URL")
    if env_urls:
        urls = [u.strip() for u in env_urls.split(",") if u.strip()]

    if not h5_path.exists() and not raw_bin.exists():
        if urls:
            _ensure_http_file_multi(urls, h5_path, download=download)
        elif download:
            _ensure_gdrive_folder(default_folder_id, raw_dir, download=True)
        else:
            raise FileNotFoundError(
                "MSONG raw file not found. Expected data/msong/msong-420-angular.hdf5 or "
                "data/msong/data_992272_420. You can also set VECIO_MSONG_URL(S) and use "
                "--download."
            )

    if h5_path.exists():
        _prepare_ann_hdf5_dataset(
            "msong",
            raw_dir,
            out_dir,
            h5_filename="msong-420-angular.hdf5",
            urls=urls,
            chunk_rows=chunk_rows,
            download=False,
            target_rows=target_rows,
            slice_seed=slice_seed,
            metric="angular",
            display_name="MSONG",
            normalize=True,
            sample_mode="first",
        )
        return

    if not raw_bin.exists():
        raise FileNotFoundError(
            f"MSONG download completed but neither {h5_path.name} nor {raw_bin.name} was found."
        )

    total_rows, dim = _read_bin_header(raw_bin)
    nb, d0 = convert_bin_to_npy(raw_bin, out_dir / "base.npy", chunk_rows=chunk_rows)
    if d0 != dim:
        raise ValueError(f"MSONG dim mismatch after conversion: header={dim}, npy={d0}")

    # Use a reproducible random query subset from the same pool. This mirrors the
    # competition-style raw source, which does not ship a separate 10k query set.
    nq = min(10_000, total_rows)
    query_idx = _sample_row_indices(total_rows, nq, int(slice_seed) + 17)
    base_mmap = np.load(out_dir / "base.npy", mmap_mode="r")
    queries = _open_out_npy(out_dir / "queries.npy", (nq, d0), np.float32)
    written = 0
    while written < nq:
        idx_chunk = query_idx[written:written + chunk_rows]
        queries[written:written + len(idx_chunk)] = np.asarray(base_mmap[idx_chunk], dtype=np.float32)
        written += len(idx_chunk)

    # Angular metric: normalize both base and query vectors in-place.
    base_rw = np.load(out_dir / "base.npy", mmap_mode="r+")
    for start in range(0, nb, chunk_rows):
        stop = min(start + chunk_rows, nb)
        arr = np.asarray(base_rw[start:stop], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        base_rw[start:stop] = arr / norms
    queries.flush()
    q_rw = np.load(out_dir / "queries.npy", mmap_mode="r+")
    for start in range(0, nq, chunk_rows):
        stop = min(start + chunk_rows, nq)
        arr = np.asarray(q_rw[start:stop], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        q_rw[start:stop] = arr / norms

    meta = {
        "dataset": "msong",
        "format": "clivf-npy-v1",
        "metric": "angular",
        "normalization": {"l2_unit": True},
        "base": {"rows": nb, "dim": d0, "dtype": "float32", "path": "base.npy"},
        "queries": {"rows": nq, "dim": d0, "dtype": "float32", "path": "queries.npy"},
        "groundtruth": None,
        "source": {
            "raw_dir": str(raw_dir),
            "raw_file": raw_bin.name,
            "gdrive_folder_id": default_folder_id,
            "full_rows": int(total_rows),
        },
        "query_sample": {"mode": "random-from-base", "seed": int(slice_seed) + 17, "rows": int(nq)},
    }
    _write_meta(out_dir, meta)


def prepare_random1m128(out_dir: Path, chunk_rows: int, download: bool) -> None:
    del download
    _prepare_random_dataset(
        "random1m128",
        out_dir,
        nb=1_000_000,
        nq=10_000,
        dim=128,
        seed=12345,
        chunk_rows=chunk_rows,
    )


PREPARERS = {
    "sift1m": prepare_sift1m,
    "sift10m": prepare_sift10m,
    "sift100m": prepare_sift100m,
    "gist1m": prepare_gist1m,
    "deep1m": prepare_deep1m,
    "deep1b": prepare_deep1b,
    "deep10m": prepare_deep10m,
    "deep100m": prepare_deep100m,
    "msong": prepare_msong,
    "openimage": prepare_openimage,
    "glove": prepare_glove,
    "random1m128": prepare_random1m128,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare datasets into unified .npy format")
    ap.add_argument("--dataset", "-d", required=True, choices=sorted(PREPARERS.keys()),
                    help="Dataset key to prepare")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: data/processed/<dataset>)")
    ap.add_argument("--chunk-rows", type=int, default=65536,
                    help="Rows per chunk when converting")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing processed outputs")
    ap.add_argument("--download", action="store_true",
                    help="Download raw dataset files automatically when missing")
    return ap.parse_args()


def _validate_array_record(
    out_dir: Path,
    record: dict,
    expected: dict,
    *,
    section: str,
    path_key: str = "path",
    dim_key: str = "dim",
) -> None:
    relative = record.get(path_key)
    if relative != expected["path"]:
        raise ValueError(
            f"metadata {section}.{path_key} differs: {relative!r} != {expected['path']!r}"
        )
    path = (out_dir / relative).resolve()
    if path.parent != out_dir.resolve():
        raise ValueError(f"metadata {section}.{path_key} escapes output directory")
    rows = int(record["rows"])
    allowed_rows = [int(value) for value in expected["rows"]]
    if rows not in allowed_rows:
        raise ValueError(f"metadata {section}.rows differs: {rows} not in {allowed_rows}")
    dimension = int(record[dim_key])
    if dimension != int(expected["dim"]):
        raise ValueError(
            f"metadata {section}.{dim_key} differs: {dimension} != {expected['dim']}"
        )
    declared_dtype = record.get("dtype")
    if declared_dtype is not None and str(declared_dtype) != str(expected["dtype"]):
        raise ValueError(
            f"metadata {section}.dtype differs: {declared_dtype!r} != {expected['dtype']!r}"
        )
    array = np.load(path, mmap_mode="r")
    expected_shape = (rows, dimension)
    if array.shape != expected_shape:
        raise ValueError(f"{section} shape differs: {array.shape} != {expected_shape}")
    if str(array.dtype) != str(expected["dtype"]):
        raise ValueError(f"{section} dtype differs: {array.dtype} != {expected['dtype']}")


def _validate_prepared_output(out_dir: Path, dataset: str) -> None:
    """Validate canonical geometry and required files before using public arrays."""
    meta_path = out_dir / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = DATASET_OUTPUT_CONTRACTS[dataset]
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"metadata dataset differs: {payload.get('dataset')!r} != {dataset!r}"
        )
    if payload.get("format") != "clivf-npy-v1":
        raise ValueError("metadata format must be 'clivf-npy-v1'")
    if payload.get("metric") != expected["metric"]:
        raise ValueError(
            f"metadata metric differs: {payload.get('metric')!r} != {expected['metric']!r}"
        )
    if expected.get("normalized") and payload.get("normalization") != {"l2_unit": True}:
        raise ValueError("metadata must declare l2_unit normalization")
    for section in ("base", "queries"):
        record = payload.get(section)
        if not isinstance(record, dict):
            raise ValueError(f"metadata has no {section} record")
        _validate_array_record(out_dir, record, expected[section], section=section)
    expected_groundtruth = expected.get("groundtruth")
    if expected_groundtruth is not None:
        record = payload.get("groundtruth")
        if not isinstance(record, dict):
            raise ValueError("metadata has no required groundtruth record")
        _validate_array_record(
            out_dir,
            record,
            expected_groundtruth,
            section="groundtruth",
            path_key="I_path",
            dim_key="k",
        )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _default_processed_dir(args.dataset)

    # A metadata-only directory contains no loadable dataset. Treat it as
    # empty so interrupted or older preparations can be retried without
    # requiring --force.
    BIG_OUTPUTS = ("base.npy", "queries.npy", "gt_I.npy", "gt_D.npy")
    STUB_NAMES = {"meta.json"}
    if out_dir.exists():
        contents = {p.name for p in out_dir.iterdir()}
        complete = all((out_dir / name).is_file() for name in ("meta.json", "base.npy", "queries.npy"))
        if complete and not args.force:
            try:
                _validate_prepared_output(out_dir, args.dataset)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"Existing processed dataset is invalid: {out_dir}: {exc}; "
                    "inspect it or use --force to rebuild"
                ) from None
            print(f"[skip] public dataset '{args.dataset}' is already prepared at: {out_dir}")
            return 0
        has_payload = any((out_dir / name).exists() for name in BIG_OUTPUTS)
        non_stub_extra = contents - STUB_NAMES
        if has_payload or non_stub_extra:
            if not args.force:
                raise SystemExit(f"Output dir not empty: {out_dir} (use --force to overwrite)")
            for name in ("meta.json", *BIG_OUTPUTS):
                p = out_dir / name
                if p.exists():
                    p.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    PREPARERS[args.dataset](out_dir, chunk_rows=args.chunk_rows, download=args.download)
    _validate_prepared_output(out_dir, args.dataset)
    print(f"Prepared dataset '{args.dataset}' at: {out_dir}")
    return 0


if __name__ == "__main__":
    # Ensure cwd is repo root for relative data paths.
    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(main())

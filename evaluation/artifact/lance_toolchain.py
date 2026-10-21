#!/usr/bin/env python3
"""Build and verify companion Lance examples owned by paper campaigns."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from .campaigns import (
    FIGURE11_REQUIRED_CAPABILITIES,
    LANCE_BASE_COMMIT,
    LANCE_EXPORTED_SOURCE_SHA256,
)
from .environment import discover_lance_root


FIGURE11_SOURCES = {
    "strict_budget_real_s3": (
        "rust/lance-io/examples/strict_budget_real_s3.rs",
        ("only_method",),
    ),
    "full_pipeline_real_s3": (
        "rust/lance/examples/full_pipeline_real_s3.rs",
        ("--out-query-ranges-json",),
    ),
}

LANCE_BUILD_SOURCE_PATHS = (
    ".cargo",
    "Cargo.lock",
    "Cargo.toml",
    "python",
    "protos",
    "rust",
    "rust-toolchain.toml",
)

# Lance index tests leave ignored runtime fixtures below this directory.  They
# are not Cargo inputs and must not make an otherwise exact checkout fail the
# source-export identity gate.
LANCE_NON_SOURCE_PREFIXES = ("rust/lance-index/tmp/",)
# Maturin's editable install places the compiled binding beside the Python sources.
LANCE_GENERATED_FILE_PATHS = {"python/python/lance/lance.abi3.so"}
LANCE_GENERATED_DIRECTORY_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "target",
}


def _is_non_source(relative: Path) -> bool:
    rendered = relative.as_posix()
    return (
        rendered in LANCE_GENERATED_FILE_PATHS
        or any(rendered.startswith(prefix) for prefix in LANCE_NON_SOURCE_PREFIXES)
        or any(
            part in LANCE_GENERATED_DIRECTORY_NAMES or part.endswith(".egg-info")
            for part in relative.parts
        )
    )


def _file_sha256(path: Path, relative: str) -> bytes:
    digest = hashlib.sha256()
    content = path.read_bytes()
    if relative == "Cargo.toml":
        # Anonymous GitHub rewrites the private source repository URL in text
        # exports.  This workspace metadata is not consumed by the paper
        # binaries (the source tree contains no CARGO_PKG_REPOSITORY reads), so
        # canonicalize only this field while keeping every build input pinned.
        content = re.sub(
            rb'(?m)^repository\s*=\s*"[^"]*"\s*$',
            b'repository = "<source-repository>"',
            content,
        )
    digest.update(content)
    return digest.digest()


def source_tree_sha256(lance_root: Path) -> str:
    """Hash the complete source closure used by the paper Rust binaries."""
    lance_root = lance_root.expanduser().resolve()
    files: list[Path] = []
    for relative in LANCE_BUILD_SOURCE_PATHS:
        path = lance_root / relative
        if not path.exists():
            raise ValueError(f"paper Lance source input is missing: {path}")
        if path.is_file() or path.is_symlink():
            files.append(path)
        else:
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if (candidate.is_file() or candidate.is_symlink())
                and not _is_non_source(candidate.relative_to(lance_root))
            )

    digest = hashlib.sha256()
    key = lambda item: item.relative_to(lance_root).as_posix()
    for path in sorted(files, key=key):
        relative = path.relative_to(lance_root).as_posix().encode("utf-8")
        if path.is_symlink():
            kind = b"link"
            content_digest = hashlib.sha256(os.readlink(path).encode("utf-8")).digest()
        else:
            kind = b"file"
            content_digest = _file_sha256(path, relative.decode("utf-8"))
        digest.update(kind + b"\0" + relative + b"\0" + content_digest)
    return digest.hexdigest()


def validate_lance_source(lance_root: Path) -> dict[str, str]:
    """Validate either the exact Git checkout or its anonymous source export."""
    lance_root = lance_root.expanduser().resolve()
    root_result = subprocess.run(
        ["git", "-C", str(lance_root), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    git_root = (
        Path(root_result.stdout.strip()).resolve()
        if root_result.returncode == 0 and root_result.stdout.strip()
        else None
    )
    if git_root == lance_root:
        result = subprocess.run(
            ["git", "-C", str(lance_root), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        observed = result.stdout.strip() if result.returncode == 0 else ""
        if observed != LANCE_BASE_COMMIT:
            raise SystemExit(
                "the companion Lance checkout is not at the public paper release "
                f"{LANCE_BASE_COMMIT} (observed {observed}); "
                "check out the pinned WaRP Lance commit"
            )
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            status = subprocess.run(
                ["git", "-C", str(lance_root), *args],
                check=False,
            )
            if status.returncode != 0:
                raise SystemExit("the companion Lance checkout has tracked source changes")
        try:
            fingerprint = source_tree_sha256(lance_root)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if fingerprint != LANCE_EXPORTED_SOURCE_SHA256:
            raise SystemExit(
                "the companion Lance checkout has untracked or ignored build-source "
                f"changes (observed source fingerprint {fingerprint})"
            )
        source_mode = "git-checkout"
    else:
        source_mode = "exported-source"
        try:
            fingerprint = source_tree_sha256(lance_root)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if fingerprint != LANCE_EXPORTED_SOURCE_SHA256:
            raise SystemExit(
                "the companion Lance exported source tree does not match the public "
                f"paper release {LANCE_BASE_COMMIT} (observed {fingerprint})"
            )
    return {
        "lance_commit": LANCE_BASE_COMMIT,
        "source_mode": source_mode,
        "source_tree_sha256": fingerprint,
    }


def companion_root_for_binary(binary: Path) -> Path:
    """Resolve a paper binary to either its Git checkout or exported source root."""
    binary = binary.expanduser().resolve()
    configured = str(os.environ.get("VECIO_LANCE_ROOT", "")).strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / "Cargo.toml").is_file():
            return root
    for parent in binary.parents:
        if (
            (parent / "Cargo.toml").is_file()
            and (parent / "Cargo.lock").is_file()
            and (parent / "rust").is_dir()
        ):
            return parent
    result = subprocess.run(
        ["git", "-C", str(binary.parent), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).resolve()
    raise ValueError(
        "cannot resolve the paper Lance source root from the binary; "
        "set VECIO_LANCE_ROOT"
    )


def source_capability_errors(lance_root: Path) -> list[str]:
    """Inspect source only; safe for a no-build server preflight."""
    errors: list[str] = []
    for example, (relative, needles) in FIGURE11_SOURCES.items():
        path = lance_root / relative
        if not path.is_file():
            errors.append(f"{example} source is missing: {path}")
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            if needle not in source:
                errors.append(f"{example} source lacks required capability {needle!r}: {path}")
    return errors


def figure11_build_commands(lance_root: Path, target_dir: Path) -> list[list[str]]:
    manifest = lance_root / "Cargo.toml"
    return [
        [
            "cargo", "build", "--manifest-path", str(manifest), "--release",
            "-p", "lance-io", "--example", "strict_budget_real_s3",
            "--target-dir", str(target_dir),
        ],
        [
            "cargo", "build", "--manifest-path", str(manifest), "--release",
            "-p", "lance", "--example", "full_pipeline_real_s3",
            "--target-dir", str(target_dir),
        ],
    ]


def _binary_has_flag(path: Path, flag: str) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        result = subprocess.run(
            [str(path), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return flag in result.stdout


def _binary_capability_errors(target_dir: Path) -> list[str]:
    errors = []
    for example, flags in FIGURE11_REQUIRED_CAPABILITIES.items():
        binary = target_dir / "release" / "examples" / example
        for flag in flags:
            if not _binary_has_flag(binary, flag):
                errors.append(f"built binary lacks {flag}: {binary}")
    return errors


def verify_lance_history(lance_root: Path) -> dict[str, str]:
    """Require the exact paper checkout or its byte-identified source export."""
    return validate_lance_source(lance_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure11", action="store_true", help="Prepare Figure 11 examples.")
    parser.add_argument("--lance-root", default="")
    parser.add_argument("--target-dir", default="")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify previously built binaries without invoking Cargo.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Inspect source and print commands; never run Cargo or a benchmark.",
    )
    args = parser.parse_args(argv)
    if not args.figure11:
        raise SystemExit("no companion toolchain selected (expected --figure11)")
    lance_root = discover_lance_root(args.lance_root)
    if lance_root is None:
        raise SystemExit("companion Lance checkout not found; set VECIO_LANCE_ROOT or --lance-root")
    target_dir = (
        Path(args.target_dir).expanduser().resolve()
        if args.target_dir
        else lance_root / "target"
    )
    errors = source_capability_errors(lance_root)
    if errors:
        raise SystemExit(
            "Figure 11 companion source preflight failed:\n  - " + "\n  - ".join(errors)
        )
    verify_lance_history(lance_root)
    commands = figure11_build_commands(lance_root, target_dir)
    print(f"[contract] lance_base_commit={LANCE_BASE_COMMIT}")
    for command in commands:
        print("$ " + shlex.join(command))
    if args.check_only:
        print("[ok] Figure 11 source capabilities and build plan are valid; Cargo was not run")
        return 0
    if args.verify_existing:
        errors = _binary_capability_errors(target_dir)
        if errors:
            raise SystemExit(
                "Figure 11 existing binary verification failed:\n  - "
                + "\n  - ".join(errors)
            )
        print(f"[ok] existing Figure 11 companion binaries verified under {target_dir}")
        return 0
    if shutil.which("cargo") is None:
        raise SystemExit("cargo is required; run scripts/setup_ec2.sh --with-cargo")
    for command in commands:
        subprocess.run(command, cwd=lance_root, check=True)
    errors = _binary_capability_errors(target_dir)
    if errors:
        raise SystemExit(
            "Figure 11 companion binary verification failed:\n  - " + "\n  - ".join(errors)
        )
    print(f"[ok] Figure 11 companion binaries are ready under {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

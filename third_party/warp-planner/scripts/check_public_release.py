#!/usr/bin/env python3
"""Reject private metadata and credential material from a release tree."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILE_MANIFEST = ROOT / "PUBLIC_RELEASE_FILES.txt"
RELEASE_IDENTITY = "WaRP Authors <wrp-wrp@users.noreply.github.com>"
SENSITIVE_NAMES = {
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem"}
ALLOWED_EMAILS = {"wrp-wrp@users.noreply.github.com"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PERSONAL_PATH_RE = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")
PRIVATE_HOST_RE = re.compile(r"\bip-(?:[0-9]+-){3}[0-9]+\.[A-Za-z0-9.-]*internal\b")
PRIVATE_REPOSITORY_RE = re.compile(r"github[.]com/wrp-wrp/")
ACCESS_KEY_RE = re.compile(r"\bA(?:KIA|SIA)[A-Z0-9]{16}\b")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
SECRET_ASSIGNMENT_RE = re.compile(
    r"\bAWS_(?:SECRET_ACCESS_KEY|SESSION_TOKEN)\s*=\s*[^\s<][^\s]*"
)
GENERIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)^\s*(?:export\s+)?(?:[A-Z][A-Z0-9_]*_)?"
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY)\s*[:=]\s*"
    r"[\"']?(?![<$])[A-Za-z0-9/+_.=-]{8,}"
)
PRIVATE_RESOURCE_ASSIGNMENT_RE = re.compile(
    r"(?im)^\s*(?:export\s+)?(?:[A-Z][A-Z0-9_]*_)?"
    r"(?:BUCKET|ENDPOINT|HOST|URI)\s*[:=]\s*"
    r"[\"']?(?![<$])[A-Za-z][A-Za-z0-9+._:/-]{3,}"
)
SERVICE_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _exact_git_root() -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    root = Path(result.stdout.strip()).resolve()
    return root if root == ROOT else None


def _manifest_relative_paths() -> list[Path]:
    if not PUBLIC_FILE_MANIFEST.is_file():
        raise SystemExit(
            "source export has no Git metadata and PUBLIC_RELEASE_FILES.txt is missing"
        )
    relative_paths: list[Path] = []
    for line_number, raw in enumerate(
        PUBLIC_FILE_MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(
                f"PUBLIC_RELEASE_FILES.txt:{line_number}: unsafe path {value!r}"
            )
        relative_paths.append(relative)
    if relative_paths != sorted(set(relative_paths)):
        raise SystemExit("PUBLIC_RELEASE_FILES.txt must contain sorted unique paths")
    return relative_paths


def _tracked_paths() -> list[Path]:
    if _exact_git_root() is not None:
        relative_paths = sorted(
            Path(item) for item in _git("ls-files", "-z").split("\0") if item
        )
        if PUBLIC_FILE_MANIFEST.is_file():
            declared = _manifest_relative_paths()
            if relative_paths != declared:
                raise SystemExit(
                    "PUBLIC_RELEASE_FILES.txt does not match the tracked release tree"
                )
    else:
        relative_paths = _manifest_relative_paths()

    paths: list[Path] = []
    for relative in relative_paths:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"public release file is missing: {relative.as_posix()}")
        paths.append(path)
    return sorted(paths)


def _text(path: Path) -> str | None:
    payload = path.read_bytes()
    if b"\0" in payload:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _content_violations(location: str, text: str) -> list[str]:
    violations: list[str] = []
    for match in EMAIL_RE.finditer(text):
        if match.group(0).lower() not in ALLOWED_EMAILS:
            line = text.count("\n", 0, match.start()) + 1
            violations.append(f"email address: {location}:{line}")
    for label, pattern in (
        ("personal absolute path", PERSONAL_PATH_RE),
        ("private cloud hostname", PRIVATE_HOST_RE),
        ("private GitHub repository URL", PRIVATE_REPOSITORY_RE),
        ("access key identifier", ACCESS_KEY_RE),
        ("private key", PRIVATE_KEY_RE),
        ("inline AWS secret", SECRET_ASSIGNMENT_RE),
        ("generic secret assignment", GENERIC_SECRET_ASSIGNMENT_RE),
        ("private resource assignment", PRIVATE_RESOURCE_ASSIGNMENT_RE),
        ("service access token", SERVICE_TOKEN_RE),
    ):
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            violations.append(f"{label}: {location}:{line}")
    return violations


def tree_violations() -> list[str]:
    violations: list[str] = []
    for path in _tracked_paths():
        relative = path.relative_to(ROOT)
        lowered = path.name.lower()
        if (
            lowered == ".env"
            or lowered.startswith(".env.")
            or lowered in SENSITIVE_NAMES
            or path.suffix.lower() in SENSITIVE_SUFFIXES
        ):
            violations.append(f"sensitive tracked filename: {relative}")
        text = _text(path)
        if text is None:
            continue
        violations.extend(_content_violations(str(relative), text))
    return sorted(set(violations))


def history_violations() -> list[str]:
    if _exact_git_root() is None:
        return ["Git history is unavailable in this source export"]
    violations: list[str] = []
    if _git("status", "--porcelain").strip():
        violations.append("release worktree is not clean")
    if _git("rev-list", "--count", "HEAD").strip() != "1":
        violations.append("release history must contain exactly one root commit")
    metadata = _git("show", "-s", "--format=%P%n%an <%ae>%n%cn <%ce>", "HEAD").splitlines()
    if len(metadata) != 3 or metadata[0]:
        violations.append("release HEAD must be a root commit")
    else:
        if metadata[1] != RELEASE_IDENTITY:
            violations.append("release author identity is not WaRP Authors noreply")
        if metadata[2] != RELEASE_IDENTITY:
            violations.append("release committer identity is not WaRP Authors noreply")
    message = _git("log", "-1", "--format=%B", "HEAD")
    violations.extend(_content_violations("release commit message", message))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also require a single clean-root WaRP Authors release commit",
    )
    args = parser.parse_args()
    violations = tree_violations()
    if args.history:
        violations.extend(history_violations())
    if violations:
        raise SystemExit("public release hygiene failed:\n  - " + "\n  - ".join(violations))
    suffix = " and sanitized release history" if args.history else ""
    print(f"[ok] no detected private metadata or credentials{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reject private metadata and credential material from the public artifact."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_FILE_MANIFEST = ROOT / "PUBLIC_RELEASE_FILES.txt"
PUBLIC_GIT_NAME = "WaRP Authors"
PUBLIC_GIT_EMAIL = "wrp-wrp@users.noreply.github.com"
SENSITIVE_FILE_SUFFIXES = {".key", ".kdbx", ".p12", ".pem", ".pfx"}
SENSITIVE_FILENAMES = {".env", "credentials", "credentials.json"}
PRIVATE_WORKFLOW_PATH = re.compile(
    r"(?:^|/)revision/(?:artifacts|figures/(?:review|approved))(?:/|$)",
    re.IGNORECASE,
)
PERSONAL_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9._-]+/"
)
S3_URI = re.compile(r"s3:[/][/]([^/\s\"',]+)")
PUBLIC_BUCKET_PLACEHOLDER = re.compile(
    r"(?:\$[A-Z_][A-Z0-9_]*|\$\{[A-Z_][A-Z0-9_]*\}|%s|"
    r"\{[A-Za-z_][A-Za-z0-9_]*\}|<[^>]+>|bucket|other|example|"
    r"example-bucket|paper-bucket|paper-artifact|your-(?:s3-)?bucket(?:-name)?)"
)
HARDCODED_BUCKET_DEFAULT = re.compile(
    r"add_argument\([^\n]*--(?:s3-)?bucket[^\n]*default\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
PRIVATE_CLOUD_IDENTITY = re.compile(
    r"(?:\bi-[0-9a-f]{8,17}\b|\bip-[0-9-]+\.[A-Za-z0-9.-]*compute\.internal\b)"
)
PRIVATE_REPOSITORY_URL = re.compile(r"github[.]com/wrp-wrp/([A-Za-z0-9_.-]+)")
PUBLIC_REPOSITORIES = {"warp-artifact", "warp-artifact.git"}
EMAIL_ADDRESS = re.compile(
    r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
CREDENTIAL_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\baws_(?:access_key_id|secret_access_key|session_token)\s*[:=]", re.IGNORECASE),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r'''(?:_authToken|api[_-]?key|client[_-]?secret)\s*[:=]\s*["'][^"']+["']''',
        re.IGNORECASE,
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
OPAQUE_REVISION_ID = re.compile(
    r"(?<![A-Za-z0-9])F\d{3}(?:-(?:S|D)\d+)?(?:-[A-Z0-9][A-Z0-9-]*)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
OPAQUE_STAGE_ID = re.compile(
    r"(?<![A-Za-z0-9])(?:S|D)\d{2,3}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PHYSICAL_OBJECT_NAME = re.compile(
    r"\b(?:s\d{2,3}_[a-z0-9_]+|"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\.lance\b"
)
NOQA_CLAUSE = re.compile(
    r"#\s*noqa:\s*[A-Z]{1,4}\d{3}(?:\s*,\s*[A-Z]{1,4}\d{3})*"
)
FORBIDDEN_DOCUMENT_NAMES = {
    "DECISION.md",
    "DECISIONS.md",
    "PROGRESS.md",
    "PROTOCOL.md",
    "STATUS.md",
}


def _exact_git_root() -> Path | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    root = Path(completed.stdout.strip()).resolve()
    return root if root == ROOT else None


def _manifest_relative_paths() -> list[Path]:
    if not PUBLIC_FILE_MANIFEST.is_file():
        raise RuntimeError(
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
            raise RuntimeError(
                f"PUBLIC_RELEASE_FILES.txt:{line_number}: unsafe path {value!r}"
            )
        relative_paths.append(relative)
    if relative_paths != sorted(set(relative_paths)):
        raise RuntimeError("PUBLIC_RELEASE_FILES.txt must contain sorted unique paths")
    return relative_paths


def tracked_files() -> list[Path]:
    git_root = _exact_git_root()
    if git_root is not None:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        relative_paths = sorted(
            Path(raw.decode("utf-8"))
            for raw in completed.stdout.split(b"\0")
            if raw
        )
        if PUBLIC_FILE_MANIFEST.is_file():
            declared = _manifest_relative_paths()
            if relative_paths != declared:
                raise RuntimeError(
                    "PUBLIC_RELEASE_FILES.txt does not match the tracked release tree"
                )
    else:
        relative_paths = _manifest_relative_paths()

    paths: list[Path] = []
    for relative in relative_paths:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"public release file is missing: {relative.as_posix()}")
        paths.append(path)
    return sorted(paths)


def _read_text_if_probable(path: Path) -> str | None:
    """Read any tracked text payload, regardless of its filename or suffix."""
    payload = path.read_bytes()
    if b"\0" in payload:
        return None
    return payload.decode("utf-8", errors="replace")


_UNRESOLVED = object()


def _static_value(node: ast.AST, names: dict[str, object]) -> object:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id, _UNRESOLVED)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_static_value(value, names) for value in node.elts]
        if any(value is _UNRESOLVED for value in values):
            return _UNRESOLVED
        return values
    if isinstance(node, ast.BinOp):
        left = _static_value(node.left, names)
        right = _static_value(node.right, names)
        if left is _UNRESOLVED or right is _UNRESOLVED:
            return _UNRESOLVED
        if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(node.op, ast.Mult):
            if isinstance(left, str) and isinstance(right, int):
                return left * right
            if isinstance(left, int) and isinstance(right, str):
                return right * left
        return _UNRESOLVED
    if isinstance(node, ast.FormattedValue):
        value = _static_value(node.value, names)
        if value is not _UNRESOLVED and isinstance(value, (str, int)):
            return str(value)
        return _UNRESOLVED
    if isinstance(node, ast.JoinedStr):
        parts = [_static_value(value, names) for value in node.values]
        if any(part is _UNRESOLVED or not isinstance(part, str) for part in parts):
            return _UNRESOLVED
        return "".join(parts)
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "chr"
            and len(node.args) == 1
            and not node.keywords
        ):
            value = _static_value(node.args[0], names)
            if isinstance(value, int):
                try:
                    return chr(value)
                except ValueError:
                    return _UNRESOLVED
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
            and not node.keywords
        ):
            value = _static_value(node.args[0], names)
            if value is not _UNRESOLVED and isinstance(value, (str, int)):
                return str(value)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and len(node.args) == 1
            and not node.keywords
        ):
            separator = _static_value(node.func.value, names)
            values = _static_value(node.args[0], names)
            if (
                isinstance(separator, str)
                and isinstance(values, list)
                and all(isinstance(value, str) for value in values)
            ):
                return separator.join(values)
    return _UNRESOLVED


class _StaticStringResolver(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[dict[str, object]] = [{}]
        self.resolved: list[tuple[int, str]] = []

    @property
    def scope(self) -> dict[str, object]:
        return self.names[-1]

    def _record(self, node: ast.AST) -> None:
        value = _static_value(node, self.scope)
        if isinstance(value, str):
            item = (getattr(node, "lineno", 0), value)
            if item not in self.resolved:
                self.resolved.append(item)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.generic_visit(node)
        self._record(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.generic_visit(node)
        self._record(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        self._record(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = _static_value(node.value, self.scope)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scope[target.id] = value

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.scope[node.target.id] = _static_value(node.value, self.scope)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        child = dict(self.scope)
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            child.pop(argument.arg, None)
        if node.args.vararg:
            child.pop(node.args.vararg.arg, None)
        if node.args.kwarg:
            child.pop(node.args.kwarg.arg, None)
        self.names.append(child)
        for statement in node.body:
            self.visit(statement)
        self.names.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def _resolved_static_strings(source: str) -> list[tuple[int, str]]:
    """Resolve common constant constructions that line scanning cannot see."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    resolver = _StaticStringResolver()
    resolver.visit(tree)
    return resolver.resolved


def _is_physical_object_identifier(line: str, match: re.Match[str]) -> bool:
    return any(
        candidate.start() <= match.start() and match.end() <= candidate.end()
        for candidate in PHYSICAL_OBJECT_NAME.finditer(line)
    )


def _is_linter_suppression_identifier(line: str, match: re.Match[str]) -> bool:
    return any(
        candidate.start() <= match.start() and match.end() <= candidate.end()
        for candidate in NOQA_CLAUSE.finditer(line)
    )


def _unallowlisted_matches(pattern: re.Pattern[str], text: str) -> list[re.Match[str]]:
    return [
        match
        for match in pattern.finditer(text)
        if not _is_physical_object_identifier(text, match)
        and not _is_linter_suppression_identifier(text, match)
    ]


def _privacy_violations(text: str) -> list[str]:
    errors: list[str] = []
    if PERSONAL_ABSOLUTE_PATH.search(text):
        errors.append("personal absolute path")
    for match in S3_URI.finditer(text):
        if not PUBLIC_BUCKET_PLACEHOLDER.fullmatch(match.group(1)):
            errors.append("non-placeholder S3 bucket")
    for match in HARDCODED_BUCKET_DEFAULT.finditer(text):
        if not PUBLIC_BUCKET_PLACEHOLDER.fullmatch(match.group(1)):
            errors.append("hard-coded bucket default")
    if PRIVATE_CLOUD_IDENTITY.search(text):
        errors.append("private cloud host or instance identity")
    if any(match.group(1) not in PUBLIC_REPOSITORIES
           for match in PRIVATE_REPOSITORY_URL.finditer(text)):
        errors.append("private GitHub repository URL")
    if any(match.group(0) != PUBLIC_GIT_EMAIL for match in EMAIL_ADDRESS.finditer(text)):
        errors.append("email address")
    if any(pattern.search(text) for pattern in CREDENTIAL_PATTERNS):
        errors.append("credential or private-key material")
    return errors


def violations() -> list[str]:
    errors: list[str] = []
    consolidated = (ROOT / "SOURCE_MANIFEST.json").is_file()
    if consolidated:
        from .source_integrity import verify
        errors.extend(verify(ROOT))
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if consolidated and relative.startswith("third_party/"):
            # Complete upstream snapshots are checked byte-for-byte above.
            # Public contacts and upstream test fixtures are outside the
            # first-party anonymity and private-workflow heuristics.
            continue
        if path.name.upper() in FORBIDDEN_DOCUMENT_NAMES:
            errors.append(f"{relative}: internal workflow document is tracked")
        if (
            path.name.lower() in SENSITIVE_FILENAMES
            or path.suffix.lower() in SENSITIVE_FILE_SUFFIXES
        ):
            errors.append(f"{relative}: sensitive credential filename is tracked")
        if PRIVATE_WORKFLOW_PATH.search(relative):
            errors.append(f"{relative}: private workflow path is tracked")
        for label in _privacy_violations(relative):
            errors.append(f"{relative}: path contains {label}")
        for revision_match in _unallowlisted_matches(OPAQUE_REVISION_ID, relative):
            errors.append(
                f"{relative}: path contains opaque revision id {revision_match.group(0)!r}"
            )
        for stage_match in _unallowlisted_matches(OPAQUE_STAGE_ID, relative):
            errors.append(
                f"{relative}: path contains opaque stage id {stage_match.group(0)!r}"
            )
        text = _read_text_if_probable(path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if PRIVATE_WORKFLOW_PATH.search(line):
                errors.append(f"{relative}:{line_number}: private workflow path")
            for label in _privacy_violations(line):
                errors.append(f"{relative}:{line_number}: {label}")
            for match in _unallowlisted_matches(OPAQUE_REVISION_ID, line):
                errors.append(
                    f"{relative}:{line_number}: opaque revision id {match.group(0)!r}"
                )
            for stage_match in _unallowlisted_matches(OPAQUE_STAGE_ID, line):
                errors.append(
                    f"{relative}:{line_number}: opaque stage id {stage_match.group(0)!r}"
                )
        if path.suffix.lower() == ".py":
            for line_number, value in _resolved_static_strings(text):
                if PRIVATE_WORKFLOW_PATH.search(value):
                    errors.append(
                        f"{relative}:{line_number}: resolved literal contains private workflow path"
                    )
                for label in _privacy_violations(value):
                    errors.append(
                        f"{relative}:{line_number}: resolved literal contains {label}"
                    )
                for match in _unallowlisted_matches(OPAQUE_REVISION_ID, value):
                    errors.append(
                        f"{relative}:{line_number}: resolved literal contains opaque revision id "
                        f"{match.group(0)!r}"
                    )
                for match in _unallowlisted_matches(OPAQUE_STAGE_ID, value):
                    errors.append(
                        f"{relative}:{line_number}: resolved literal contains opaque stage id "
                        f"{match.group(0)!r}"
                    )
    return errors


def history_violations() -> list[str]:
    """Require the publishable branch to be a single sanitized root commit."""
    if _exact_git_root() is None:
        return ["Git history is unavailable in this source export"]
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    errors = []
    if count != "1":
        errors.append(
            f"Git history has {count} reachable commits; public release must use one clean root commit"
        )
        return errors
    metadata = subprocess.run(
        ["git", "show", "-s", "--format=%an%n%ae%n%cn%n%ce%n%B", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(metadata) < 4:
        return ["unable to parse public release commit identity"]
    author_name, author_email, committer_name, committer_email = metadata[:4]
    if (author_name, author_email) != (PUBLIC_GIT_NAME, PUBLIC_GIT_EMAIL):
        errors.append("public release author identity is not the repository noreply identity")
    if (committer_name, committer_email) != (PUBLIC_GIT_NAME, PUBLIC_GIT_EMAIL):
        errors.append("public release committer identity is not the repository noreply identity")
    for line_number, line in enumerate(metadata[4:], start=1):
        for label in _privacy_violations(line):
            errors.append(f"public release commit message:{line_number}: {label}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also require a single sanitized public-release root commit",
    )
    args = parser.parse_args(argv)
    errors = violations()
    if args.history:
        errors.extend(history_violations())
    if errors:
        print(f"[fail] {len(errors)} public-release hygiene violation(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    suffix = " and has sanitized release history" if args.history else ""
    print(f"[ok] public artifact contains no detected private metadata or credentials{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

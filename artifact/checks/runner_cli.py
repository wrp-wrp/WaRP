#!/usr/bin/env python3
"""Static validation: every CLI flag the registry and inter-runner subprocess
calls pass to a runner is a flag that runner actually declares.

Two layers:
 1. Each spec.run command in the artifact registry.
 2. Each subprocess.run / subprocess.call inside a runner script that
    targets another runner .py file (cross-runner invocations).

Both kinds of misuse produce runtime "unrecognized arguments" errors
on EC2 minutes / hours into a full run; catch them here instead.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from ..registry import EXPERIMENTS, ROOT


def _extract_flags(runner_path: Path) -> set[str]:
    tree = ast.parse(runner_path.read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flags.add(arg.value)
    return flags


def _check_argv(argv: list, script_to_flags: dict[str, set[str]], context: str) -> list[str]:
    errors: list[str] = []
    script = None
    for part in argv:
        if isinstance(part, str) and part.endswith(".py"):
            script = part
            break
    if script is None or script not in script_to_flags:
        return errors
    declared = script_to_flags[script]
    for token in argv:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        if token in declared:
            continue
        errors.append(
            f"{context}: {script} does not declare {token} "
            f"(declared: {sorted(f for f in declared if f.startswith('--'))[:10]}...)"
        )
    return errors


def _runners_referenced_by_specs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in EXPERIMENTS:
        for cmd in spec.run.commands:
            for part in cmd.argv:
                if part.endswith(".py"):
                    p = ROOT / part
                    if p.exists():
                        out[part] = p
    # Also add every runners/*/*.py we find on disk so cross-runner calls
    # resolve.
    for p in (ROOT / "runners").rglob("*.py"):
        rel = str(p.relative_to(ROOT))
        out.setdefault(rel, p)
    return out


def _static_argv(call: ast.Call) -> list[str] | None:
    """Best-effort extract the argv list passed to subprocess.run/call."""
    if not call.args:
        return None
    arg = call.args[0]
    if not isinstance(arg, (ast.List, ast.Tuple)):
        return None
    out: list[str] = []
    for elt in arg.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
        elif isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name) and elt.func.id == "str":
            out.append("<str()>")
        else:
            out.append("<?>")
    return out


def _resolve_path_constants(tree: ast.AST) -> dict[str, str]:
    """Collect module-level assignments whose RHS looks like a path to a
    runner script, so we can resolve `str(CONST_NAME)` references later.

    Handles:
      CONST = ROOT / "runners" / "foo" / "bar.py"
      CONST = ROOT / "runners/foo/bar.py"
    """
    out: dict[str, str] = {}
    for node in tree.body if hasattr(tree, "body") else []:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        # Walk the RHS, collecting string constants from `ROOT / "a" / "b"` joins.
        parts: list[str] = []
        ok = True

        def _gather(v: ast.AST) -> None:
            nonlocal ok
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.BinOp) and isinstance(v.op, ast.Div):
                _gather(v.left)
                _gather(v.right)
            elif isinstance(v, ast.Name) and v.id in {"ROOT", "HERE"}:
                pass  # anchor; don't contribute a path segment
            else:
                ok = False

        _gather(node.value)
        if ok and parts:
            joined = "/".join(parts).replace("//", "/")
            if joined.endswith(".py") and "runners/" in joined:
                out[name] = joined
    return out


def _scan_cross_runner(caller: Path, scripts: dict[str, set[str]]) -> list[str]:
    """Look for subprocess.run([... runner.py ...]) invocations inside
    caller and check flags against the target's declared flags.

    Resolves module-level path constants like
    ``SIBLING_RUNNER = ROOT / "runners" / "foo" / "bar.py"`` so dynamic
    ``str(SIBLING_RUNNER)`` references still line up with a real target.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(caller.read_text())
    except SyntaxError:
        return errors
    path_consts = _resolve_path_constants(tree)

    def _argv_items(call: ast.Call) -> list[str | None]:
        if not call.args:
            return []
        arg = call.args[0]
        if not isinstance(arg, (ast.List, ast.Tuple)):
            return []
        items: list[str | None] = []
        for elt in arg.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                items.append(elt.value)
            elif (
                isinstance(elt, ast.Call)
                and isinstance(elt.func, ast.Name)
                and elt.func.id == "str"
                and elt.args
                and isinstance(elt.args[0], ast.Name)
                and elt.args[0].id in path_consts
            ):
                items.append(path_consts[elt.args[0].id])
            else:
                items.append(None)
        return items

    def _list_items(elts) -> list[str | None]:
        items: list[str | None] = []
        for elt in elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                items.append(elt.value)
            elif (
                isinstance(elt, ast.Call)
                and isinstance(elt.func, ast.Name)
                and elt.func.id == "str"
                and elt.args
                and isinstance(elt.args[0], ast.Name)
                and elt.args[0].id in path_consts
            ):
                items.append(path_consts[elt.args[0].id])
            else:
                items.append(None)
        return items

    # Scan every list/tuple literal in the file; if it contains a path-
    # to-another-runner, treat it as argv and validate the flags.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        items = _list_items(node.elts)
        target = None
        for item in items:
            if item and item.endswith(".py") and "runners/" in item:
                target = item
                break
        if target is None or target not in scripts:
            continue
        errors.extend(
            _check_argv(
                [it for it in items if isinstance(it, str)],
                scripts,
                context=f"{caller.relative_to(ROOT)} -> {target} (cross-runner)",
            )
        )
    return errors


def main() -> int:
    errors: list[str] = []
    runner_paths = _runners_referenced_by_specs()
    scripts: dict[str, set[str]] = {
        rel: _extract_flags(path) for rel, path in runner_paths.items()
    }

    # Layer 1: artifact registry -> runner CLI.
    for spec in EXPERIMENTS:
        for cmd in spec.run.commands:
            errors.extend(
                _check_argv(list(cmd.argv), scripts,
                            context=f"artifact registry {spec.id}")
            )

    # Layer 2: cross-runner subprocess invocations.
    for rel, path in runner_paths.items():
        errors.extend(_scan_cross_runner(path, scripts))

    # Dedupe and sort
    errors = sorted(set(errors))
    if errors:
        print(f"[fail] {len(errors)} CLI flag mismatches:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("[ok] every --flag passed to a runner (specs + cross-runner) is declared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

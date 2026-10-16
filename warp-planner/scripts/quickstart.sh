#!/usr/bin/env bash
# warp-planner quickstart.
#
# Builds the Rust crate, runs the test battery (including the golden
# regression suite), and optionally builds + tests the Python
# extension.
#
# This is a verification script, not a full environment bootstrapper:
# Rust must already be installed, and `--with-python` expects an active
# venv (or sibling `python/.venv`) with `maturin` and `pytest`
# available.
#
# Usage:  bash scripts/quickstart.sh [--with-python]
set -euo pipefail

cd "$(dirname "$0")/.."
WITH_PYTHON=0
for arg in "$@"; do
  case "$arg" in
    --with-python) WITH_PYTHON=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "[quickstart] unknown arg: $arg" >&2; exit 1 ;;
  esac
done

echo "==> cargo test --release"
cargo test --release

echo
echo "==> cargo test --release --features probe"
cargo test --release --features probe

if [[ $WITH_PYTHON -eq 1 ]]; then
  if ! command -v maturin >/dev/null 2>&1; then
    echo "error: maturin not on PATH. Install one of:"
    echo "    pip install maturin           # into the active venv"
    echo "    pipx install maturin          # isolated user-wide"
    echo "    cargo install --locked maturin"
    echo "(maturin also needs a Python virtualenv to be active.)"
    exit 1
  fi
  if ! python -m pytest --version >/dev/null 2>&1; then
    echo "error: pytest is not available to the Python interpreter."
    echo "       Activate your venv and install it with:"
    echo "         pip install pytest"
    exit 1
  fi
  echo
  echo "==> maturin develop --release (Python extension)"
  (cd python && maturin develop --release)
  echo
  echo "==> pytest python/tests"
  (cd python && python -m pytest tests -q)
fi

echo
echo "[quickstart] all green"

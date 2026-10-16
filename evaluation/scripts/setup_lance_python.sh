#!/usr/bin/env bash
# Build the Python bindings from the exact companion Lance checkout.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANCE_ROOT="${VECIO_LANCE_ROOT:-}"

usage() {
  echo "Usage: bash scripts/setup_lance_python.sh [--lance-root PATH]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lance-root) LANCE_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$LANCE_ROOT" ]]; then
  LANCE_ROOT="$(python3 - <<'PY'
from artifact.environment import discover_lance_root
root = discover_lance_root()
if root is None:
    raise SystemExit("companion Lance checkout not found; set VECIO_LANCE_ROOT")
print(root)
PY
)"
fi
LANCE_ROOT="$(cd "$LANCE_ROOT" && pwd)"
if [[ ! -f "$LANCE_ROOT/python/pyproject.toml" ]]; then
  echo "companion Lance Python package is missing: $LANCE_ROOT/python/pyproject.toml" >&2
  exit 2
fi

cd "$ROOT"
python3 runners/shared/verify_lance_base.py --lance-root "$LANCE_ROOT"
echo "[lance-python] source=$LANCE_ROOT/python"
python3 -m pip install --upgrade --break-system-packages -r requirements-full.txt 2>/dev/null \
  || python3 -m pip install --upgrade -r requirements-full.txt
python3 -m pip install --break-system-packages --editable "$LANCE_ROOT/python" 2>/dev/null \
  || python3 -m pip install --editable "$LANCE_ROOT/python"
python3 - <<'PY'
import lance
print(f"[lance-python] import ok: {lance.__file__}")
PY

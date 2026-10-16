#!/usr/bin/env bash
# Paper Experiment wrapper; registry: artifact/registry.py.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PROFILE="${1:-full}"
DATA_SLOT=latest
if [[ "$PROFILE" == "light" ]]; then
  DATA_SLOT=paper
fi
exec python3 -m artifact rebuild --only figure-04-object-store-io --profile "$PROFILE" --data-slot "$DATA_SLOT"

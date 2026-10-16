#!/usr/bin/env bash
# Regenerate the FAISS-IVF fixtures under rust/tests/data/.
#
# Most users never need to run this — the fixtures are committed. This
# script exists so that after changing `generate_fixtures.py` or
# adding a new distribution you can refresh the inputs:
#
#   bash scripts/prepare-fixtures.sh
#
# The SIFT1M distributions require a local copy of base.npy +
# queries.npy (fvecs-to-npy converted). Set SIFT1M_DIR to point at
# them; the gaussian and heavy-tail distributions are synthetic and
# need no external data.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="rust/tests/data"
GEN="$DATA_DIR/generate_fixtures.py"

if ! python3 -c "import faiss" >/dev/null 2>&1; then
    echo "[prepare-fixtures] faiss-cpu not found. Install:" >&2
    echo "    pip install faiss-cpu numpy" >&2
    exit 1
fi

echo "[1/4] gaussian (synthetic, 1k × 64)"
python3 "$GEN" --dataset gaussian --nlist 1024 --nprobe 64 \
    --num-queries 32 --seed 42 \
    --out-dir "$DATA_DIR/gaussian_1024x64"

echo "[2/4] heavy_tail (synthetic, 4k × 128)"
python3 "$GEN" --dataset heavy_tail --nlist 4096 --nprobe 128 \
    --num-queries 32 --seed 42 \
    --out-dir "$DATA_DIR/heavy_tail_4096x128"

if [[ -n "${SIFT1M_DIR:-}" && -d "$SIFT1M_DIR" ]]; then
    echo "[3/4] sift1m no-reorder (4k × 128)"
    python3 "$GEN" --dataset sift1m --dataset-dir "$SIFT1M_DIR" \
        --nlist 4096 --nprobe 128 --num-queries 32 --seed 42 \
        --reorder none \
        --out-dir "$DATA_DIR/sift_paper_4096x128"

    echo "[4/4] sift1m pca-reorder (4k × 128)"
    python3 "$GEN" --dataset sift1m --dataset-dir "$SIFT1M_DIR" \
        --nlist 4096 --nprobe 128 --num-queries 32 --seed 42 \
        --reorder pca \
        --out-dir "$DATA_DIR/sift_paper_pca_4096x128"
else
    echo "[skip] SIFT1M_DIR not set — keeping existing sift_paper_* fixtures."
    echo "       To refresh them: SIFT1M_DIR=/path/to/sift1m bash scripts/prepare-fixtures.sh"
fi

echo
echo "[prepare-fixtures] done. Next: make bless  (re-freezes expected_plans.json)."

# Top-level convenience targets — common dev loops in one command each.

PYTHON ?= python3

.PHONY: help test test-all bench bench-smoke bless python python-test clean fmt lint release-check

help:
	@echo "Targets:"
	@echo ""
	@echo "  Develop loop"
	@echo "    make test         cargo test --release (core crate, ~5s)"
	@echo "    make test-all     cargo test --release --features probe (~10s)"
	@echo "    make fmt          cargo fmt --all"
	@echo "    make lint         cargo clippy --workspace --all-targets -- -D warnings"
	@echo "    make clean        cargo clean"
	@echo ""
	@echo "  Python extension"
	@echo "    make python       cd python && maturin develop --release"
	@echo "    make python-test  pytest python/tests"
	@echo ""
	@echo "  Performance"
	@echo "    make bench-smoke  short checked-API Criterion smoke (~10s after build)"
	@echo "    make bench        full Criterion sweep, ~15-25 min, CSV under target/bench-artifacts/"
	@echo ""
	@echo "  Maintenance"
	@echo "    make bless        re-freeze golden plan snapshots after an algo change"
	@echo "    make release-check reject private metadata; add HISTORY=1 on the clean release"

test:
	cargo test --release

test-all:
	cargo test --release --features probe

bench:
	bash scripts/bench.sh

bench-smoke:
	cargo bench -p warp-planner --bench planner_smoke

bless:
	cargo run --release --example bless_golden

python:
	@command -v maturin >/dev/null 2>&1 || { \
	  echo "error: maturin not found on PATH."; \
	  echo "       Install one of:"; \
	  echo "         pip install maturin           # into the active env"; \
	  echo "         pipx install maturin          # isolated user-wide"; \
	  echo "         cargo install --locked maturin"; \
	  echo "       maturin also needs a Python virtualenv to be active"; \
	  echo "       (or a sibling python/.venv); see python/README.md."; \
	  exit 1; }
	cd python && maturin develop --release

python-test:
	@$(PYTHON) -m pytest --version >/dev/null 2>&1 || { \
	  echo "error: pytest not found. Run: pip install pytest"; exit 1; }
	cd python && $(PYTHON) -m pytest tests -q

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets -- -D warnings

release-check:
	python3 -m unittest discover -s scripts/tests -p 'test_*.py'
	python3 scripts/check_public_release.py $(if $(HISTORY),--history,)

clean:
	cargo clean

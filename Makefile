# turbo-ckf developer shortcuts.
#
# The repo standardises on a project-local virtualenv at .venv-turbo-ckf/,
# created by turbo_ckf/setup_env.sh. All targets here assume that layout.

PYTHON ?= .venv-turbo-ckf/bin/python
PIP    ?= .venv-turbo-ckf/bin/pip

.PHONY: setup build test bench lint clean

setup:
	bash turbo_ckf/setup_env.sh

build:
	$(PYTHON) -m maturin develop --release -m pyproject.toml

test:
	$(PYTHON) -m pytest turbo_ckf_tests

bench:
	$(PYTHON) turbo_ckf_tests/verify_before_after.py --steps 50000 --repeats 7 --warmup 1 --parity-steps 500

lint:
	cargo fmt --check
	cargo clippy -- -D warnings

clean:
	rm -rf target/ build/ dist/ *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +

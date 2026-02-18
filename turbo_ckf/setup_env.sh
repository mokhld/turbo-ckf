#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv-turbo-ckf"
CARGO_HOME_DIR="$ROOT_DIR/.cargo"
RUSTUP_HOME_DIR="$ROOT_DIR/.rustup"

python -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/turbo_ckf/requirements-dev.txt"

mkdir -p "$CARGO_HOME_DIR" "$RUSTUP_HOME_DIR"
export CARGO_HOME="$CARGO_HOME_DIR"
export RUSTUP_HOME="$RUSTUP_HOME_DIR"

if [ ! -x "$CARGO_HOME_DIR/bin/rustc" ]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
fi

export PATH="$CARGO_HOME_DIR/bin:$PATH"
HOST_TRIPLE="$("$CARGO_HOME_DIR/bin/rustc" -vV | awk '/^host:/ {print $2}')"
"$VENV_DIR/bin/python" -m maturin develop --release --target "$HOST_TRIPLE" --pip-path "$VENV_DIR/bin/pip"

cat <<EOF
Environment ready.

Activate virtualenv:
  source "$VENV_DIR/bin/activate"

Load rust toolchain in shell:
  export CARGO_HOME="$CARGO_HOME_DIR"
  export RUSTUP_HOME="$RUSTUP_HOME_DIR"
  source "$CARGO_HOME_DIR/env"

Run tests:
  PYTHONPATH="$ROOT_DIR" "$VENV_DIR/bin/python" -m unittest discover -s "$ROOT_DIR/turbo_ckf_tests" -p 'test_*.py'
EOF

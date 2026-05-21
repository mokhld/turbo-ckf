# Contributing

Thanks for considering a contribution to `turbo-ckf`. The project is a small
Rust + Python codebase, so the dev loop is intentionally short.

## Getting set up

```bash
git clone https://github.com/mokhld/turbo-ckf.git
cd turbo-ckf
bash turbo_ckf/setup_env.sh
```

`setup_env.sh` provisions a local `.venv-turbo-ckf/` virtualenv, installs the
Python dev requirements, and builds the Rust extension via `maturin develop`.

If you already have a Rust toolchain and just want to rebuild the extension:

```bash
.venv-turbo-ckf/bin/python -m maturin develop --release -m pyproject.toml
```

## Running tests

```bash
.venv-turbo-ckf/bin/python -m pytest turbo_ckf_tests
```

Or use the `Makefile` shortcuts:

```bash
make build   # rebuild Rust extension
make test    # run the test suite
make bench   # run verify_before_after.py end-to-end
make lint    # cargo fmt --check + cargo clippy -- -D warnings
```

## Code style

- Rust: `cargo fmt` and `cargo clippy -- -D warnings` must both pass.
- Python: keep imports sorted, prefer explicit dtypes, and add type hints to
  new public surface. `ruff` is fine if you have it installed; it is not yet
  required by CI.
- Keep public APIs documented with short docstrings explaining shape contracts.

## Adding a benchmark

New benchmarks live under `turbo_ckf_tests/` next to `benchmark.py` and
`verify_before_after.py`. Follow these rules:

- Use `time.perf_counter()` and report a median of at least 5 repeats.
- Print the platform, Python version, NumPy version, and CPU model.
- Compare against a FilterPy baseline when measuring algorithmic speedup.

## Pull request expectations

- All tests pass locally (`make test`).
- Parity with the existing CKF / FilterPy reference is preserved unless the
  change is explicitly a numerical fix - call that out in the PR description.
- New behaviour has a test. Bug fixes have a regression test.
- Update `CHANGELOG.md` under the `## [Unreleased]` section.
- Keep diffs focused; split unrelated changes into separate PRs.

## Reporting bugs

Open an issue with:

- `python --version`, `pip show turbo-ckf | grep Version`, OS, CPU.
- A minimal reproduction (ideally under 30 lines).
- Expected vs actual behaviour.

For correctness regressions, attach the input arrays as `.npy` if you can.

# CLAUDE.md

## What this package is

`turbo-ckf` (PyPI `turbo-ckf`, v0.9.0, alpha) is a Cubature Kalman Filter for Python
whose math runs in a Rust extension (PyO3 0.20, rust-numpy 0.20, nalgebra 0.32, rayon).
The API is FilterPy-shaped: set `x`, `P`, `Q`, `R`, call `predict()` / `update(z)`. It
also implements the KCKF AHRS equations from Yamagishi and Jing, arXiv:2602.12283
(quaternion attitude from gyro + accelerometer + magnetometer).

Value proposition (inferred from README/CHANGELOG, not stated in one place): a faster
FilterPy replacement (~4 us per 4-state predict+update step since v0.9.0, ~22x more for
linear sequences via `batch_filter`)
that also reports numerical health honestly (jitter counters, NIS, singular-innovation
counts), with linear batch, parallel-bank and RTS smoothing paths.

Users: Python engineers and researchers in sensor fusion, tracking and orientation
estimation who need more speed than FilterPy; Monte-Carlo users (`batch_parallel_step`);
AHRS users following the paper (its benchmarks are on M1 and Raspberry Pi 4).

## Codebase map

- `src/lib.rs` (single file, ~2.2k lines). Rust module `turbo_ckf._rust`:
  - `CubatureKalmanFilter` pyclass (~L51-1000): owns all filter state. Step methods
    `predict_custom` (Python `fx` callback), `predict_standard_model[_ckf]` (with
    `layout`), `predict_linear_model[_ckf]`, `update` (Python `hx`), `update_paper_ahrs`;
    state access `get`/`set`/`load_snapshot`/`copy`/`record_step` (~L410-510), plus the
    older `set_state`/`snapshot`.
  - Shared helpers: `cubature_points`, `stable_cholesky` (jitter relative to each diagonal
    entry, 1e-12 ... 1e-6 of it; plain factorization tried first),
    `invert_innovation` (Cholesky, then LU, then pinv), `paper_observation_model`.
  - Free functions: `rts_smooth` (~L1030), `batch_filter_linear` (~L1151),
    `batch_parallel_step` (~L1472, rayon, GIL released, status codes 0-3).
  - `SquareRootCubatureKalmanFilter` pyclass (~L1695-2224) with the same state-access
    methods, `qr_to_lower_factor` and `cholesky_downdate`.
- `turbo_ckf/core.py` (~1.9k lines): input coercion helpers, `FilterRun` + `_run_filter`
  (Python loop behind `run()`), `_AdaptiveNoiseEstimator` (residual-form R, heuristic Q),
  `_Output`/`_Factor` lazy attribute descriptors, `_BackendStateMixin` (x/P/Q/R/dt
  properties, sync, copy/to_dict/from_dict/pickle/reset), `TurboCKF` (~L764),
  `TurboSRCKF` (~L1660). `batch_filter`, `batch_parallel_step`, `rts_smooth` are static methods on
  `TurboCKF` that validate/broadcast and call the Rust free functions.
- `turbo_ckf/paper_ahrs.py`: pure-NumPy helpers for the paper's equations (F from gyro,
  Q from quaternion, R, m_N/m_D, observation model).
- `turbo_ckf/__init__.py`: public exports. `turbo_ckf/_rust.abi3.so` is a gitignored build
  artifact; rebuild it after editing `src/lib.rs`.
- `turbo_ckf_tests/`: pytest + unittest tests (`test_*.py`); `benchmark.py`,
  `benchmark_paper.py`, `verify_before_after.py` are scripts, not tests.
- `examples/quickstart_cv.py`: 1-D constant-velocity example.
- `.github/workflows/`: `ci.yml` (cargo fmt + clippy; pytest on Python 3.9-3.13 on Ubuntu
  with the 95% coverage gate, plus macOS and Windows on 3.12), `release.yml` (wheels for manylinux x86_64, macOS arm64,
  Windows x64, plus sdist; trusted publishing; tag must match both manifests),
  `security.yml` (gitleaks).

## Build, test, run

- First-time setup: `bash turbo_ckf/setup_env.sh` (creates `.venv-turbo-ckf/`, a
  repo-local Rust toolchain in `.cargo/` and `.rustup/`, then `maturin develop`).
- Rebuild the extension: `make build`. On Apple silicon, if maturin produces an x86_64
  build, use cargo directly:
  `PATH="$PWD/.cargo/bin:$PATH" RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo build --release --target aarch64-apple-darwin`
  then copy `target/aarch64-apple-darwin/release/libturbo_ckf_rust.dylib` to
  `turbo_ckf/_rust.abi3.so`.
- Tests: `PYTHONPATH=. .venv-turbo-ckf/bin/python -m pytest turbo_ckf_tests` (261 tests,
  ~8 s, 100% Python line coverage as of v0.9.0). There are no Rust unit tests (backlog B9).
- Lint: `make lint` (`cargo fmt --check`, `cargo clippy -- -D warnings`).
- Benchmarks: `make bench`, or `.venv-turbo-ckf/bin/python turbo_ckf_tests/benchmark_paper.py`.
- Working from a git worktree (for example a parallel agent): the worktree has no
  `.venv`, `.cargo` or built `.so`. Use the main checkout's toolchain and venv by absolute
  path (`CARGO_HOME`/`RUSTUP_HOME` pointing at its `.cargo`/`.rustup`), build with the
  cargo command above, copy the dylib into the worktree's `turbo_ckf/`, and run tests with
  `PYTHONPATH=$PWD`. Run ad-hoc scripts from a file or from the worktree root: `python -c`
  or `python -` puts the current directory first on `sys.path` and can import another
  checkout's package.
- Release: bump the version in `pyproject.toml`, `Cargo.toml` and `Cargo.lock`, move the
  CHANGELOG `[Unreleased]` entries under the new version, merge, tag `vX.Y.Z` on `main`,
  then create a GitHub Release from the tag. `release.yml` builds the wheels and
  publishes to PyPI on the release event; the `pypi` environment has no required
  reviewers, so this publishes immediately. `PUBLISHING.md` is gitignored (local only).

## Conventions and gotchas

- The Rust backend owns filter state. Python attributes are lazy: `x`/`P`/`Q`/`R` getters
  hand out an array tracked in `_live` (with its `tobytes()`); outputs are cached in
  `_cache`. Every wrapper call runs `_sync()` (drops the cache, pushes only handed-out
  arrays whose bytes changed) before the Rust call and `_stepped(replaced)` (drops the
  cache, marks replaced arrays read-only) after it. Assignment pushes at once via
  `backend.set(name, value)`. New backend calls must follow the same pattern.
- In-place edits of `x`/`P`/`Q`/`R` take effect on the next call; arrays replaced by a
  step or assignment are frozen so stale writes raise. Outputs never feed back.
- `TurboSRCKF` keeps `chol_P` in Rust and factors P/Q/R only on assignment (or on an
  in-place edit at the next call). Every jitter-adding `stable_cholesky` (seed, per-call
  R, downdate fallback) counts in `jitter_count`. `to_dict()` stores the factors, so
  restoring never re-factors.
- `copy()`, `from_dict()` and pickling (`__reduce__` through `to_dict()`) keep the
  counters; `reset()` zeroes them. fx/hx are pickled by reference.
- `run()` calls the backend directly and records each row with `record_step`. It falls
  back to `kf.update()` when adaptive noise is on.
- `TurboCKF` cubature moments are computed from deviations about the mean; keep it that
  way (the raw `E[x x^T] - mean mean^T` form fails for ECEF-scale states).
- Callbacks are vectorized: `fx(sigmas, dt, *fx_args)` and `hx(sigmas, *hx_args)` take a
  `(2*dim_x, dim_x)` array of row sigma points and return `(2*dim_x, dim_x)` /
  `(2*dim_x, dim_z)`. A list or tuple `fx_args` is unpacked; anything else is one arg.
- Standard models default to the blocked layout `[pos..., vel...(, acc...)]`; pass
  `layout="interleaved"` for FilterPy's `[x, vx, y, vy]`. A mismatch is not detectable.
- `update_paper_ahrs(z, sigma_acc2, sigma_mag2)` needs `dim_x=4, dim_z=6`, normalizes the
  accel and mag 3-vectors itself (sigmas are unit-vector variances), records the normalized
  `z`, and overwrites `R`. The callback path with `observation_model` does not normalize.
- `batch_filter` does predict then update at every step (so `x0`/`P0` are the state
  before the first predict) and returns posteriors. `Fs[k]` is the k-1 -> k transition in
  both `batch_filter` and `rts_smooth` (FilterPy convention); `rts_smooth` also accepts a
  length N-1 array where entry k is the k -> k+1 transition.
- `update(z=None)` is the missed-measurement path. NaN/inf raises everywhere: in `z`, in
  `batch_filter` inputs, and in `fx`/`hx` output (checked in Rust before x/P change).
  `batch_parallel_step` marks a filter with non-finite inputs as status 3 instead.
- CONTRIBUTING requires a CHANGELOG entry under `[Unreleased]` and a regression test for
  every bug fix. Keep parity with FilterPy unless a change is an intentional numerical fix.

## Review findings

Prioritized bugs and feature opportunities, with evidence, repro snippets and status, are
in `docs/REVIEW.md` (review and fix pass 2026-09-24). Update item statuses there when you
fix something. Agent-ready feature specs are in `docs/FEATURES.md`; to build the next
feature, point an agent at that file.

## Working Principles

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

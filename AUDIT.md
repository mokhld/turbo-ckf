# turbo-ckf — Repo Audit

Date: 2026-05-21
Method: Four parallel investigation agents covering (1) Rust backend, (2) Python wrapper, (3) tests & benchmarks, (4) CI/packaging/DX. Findings are merged below and grouped by **Bugs**, **Improvements**, and **Features**. Every item cites `file:line` so it can be acted on directly.

## Session 1 status (2026-05-21)

Items addressed in this pass — line references are pre-change since the codebase
has shifted. Search the new code with the diagnostic name if you want to verify.

**Rust backend (`src/lib.rs`):**
- `stable_cholesky` now returns the applied jitter; backend tracks
  `last_jitter`, `max_jitter`, `jitter_count` and exposes them via
  `snapshot()`. Singular innovations also counted (`singular_innovation_count`).
- `update_likelihood_terms` uses Cholesky log-det (sum of `log(diag(L))`)
  instead of LU determinant. Stores NIS (squared Mahalanobis) explicitly.
- Hot-path conversions rewritten: `pyarray2_to_dmatrix` uses
  `DMatrix::from_row_iterator`; `dmatrix_to_pyarray` uses a single `Vec<f64>`
  + `ndarray::Array2::from_shape_vec` instead of `Vec<Vec<f64>>`.
- `symmetrize` is now `symmetrize_in_place` (no transpose-clone allocation).
- Transpose-multiplies replaced with `tr_mul`.
- Validates `dt` finite in constructor + `predict_custom`.
- Validates `sigma_acc2`, `sigma_mag2` finite + positive in
  `update_paper_ahrs`.
- Hot helpers (`row_mean`, `symmetrize_in_place`, `cubature_points`,
  `record_jitter`) marked `#[inline]`.
- New `supported_standard_models()` static method (lets the Python wrapper
  validate names with a friendly error before reaching the backend).
- New `clear_update_diagnostics()` for the `update(z=None)` path.
- Misleading "scipy upper-triangular convention" comment rewritten.

**Python wrapper (`turbo_ckf/core.py`):**
- `update(z=None)` no longer silently leaves stale `y`, `S`, `SI`, `K`,
  `log_likelihood`, `mahalanobis`, `nis` — all are cleared via the
  backend's `clear_update_diagnostics` so a downstream NIS gate can't
  misread the previous update.
- `predict_standard_model[_ckf]` validates `model_type` against the
  backend's whitelist before calling Rust — typos surface as Python
  `ValueError` listing valid names.
- `update_paper_ahrs` validates sigmas positive + finite up front.
- `_coerce_args` now unpacks `list`-typed callback args (was
  silently wrapping them as a single positional arg).
- Better error messages on shape-mismatch callbacks (point users to the
  vectorized-callback contract).
- Added public API: `__repr__`, `reset()`, `copy()` / `__deepcopy__`,
  `to_dict()` / `from_dict()`, `gate(threshold)`.
- Added attributes: `nis`, `last_jitter`, `max_jitter`, `jitter_count`,
  `singular_innovation_count` — all pulled from each backend snapshot.
- Type hints added to all public methods.
- `paper_ahrs` helpers re-exported from `turbo_ckf` package root.
- `__init__.py` now exports the full surface.

**Tests (`turbo_ckf_tests/`):**
- New `test_correctness_invariants.py` with 16 tests covering: 10k-step
  symmetry/PSD invariants on P, NIS chi-squared sanity bounds, jitter
  counter behavior on near-singular P, `dt=0` predict identity, `dim_z
  > dim_x` shapes, scalar-R broadcast verified through math (not just
  via attribute readback), mismatched-z shape rejected,
  `update(z=None)` diagnostic clearing, `reset()` / `copy()` / `to_dict`
  round-trips, gate-acceptance/rejection, validation rejections.
- Existing `test_api.py:test_predict_update_signatures` updated to
  check parameter names + defaults instead of the full str()-of-signature
  (so adding type hints doesn't break it).
- Full suite is 51 tests, all green; benchmark suite unchanged
  (~3.3x callback / ~4.2x standard model vs FilterPy locally).

**CI / packaging / docs (sibling agent):**
- `.dockerignore` populated to exclude build artifacts, venvs, caches.
- `turbo_ckf/py.typed` shipped (PEP 561 marker).
- `pyproject.toml`: Python 3.13 classifier added; license modernized
  per PEP 639 (`license = "MIT OR Apache-2.0"` +
  `license-files = ["LICENSE-MIT", "LICENSE-APACHE"]`); `py.typed`
  included via maturin.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `Makefile` added.
- `README.md`: badges row (PyPI, Python versions, CI, license); top-level
  `pip install turbo-ckf` section.
- `.github/dependabot.yml` added (daily actions, weekly cargo + pip).
- `.github/workflows/ci.yml` split into `rust-lint` (`cargo fmt --check`
  + `cargo clippy -- -D warnings`) and `test` (Python 3.9-3.13 matrix).
- `.github/workflows/release.yml`: new `version-check` job compares
  `Cargo.toml` / `pyproject.toml` / pushed tag; build/publish jobs depend
  on it; PyPI publish has `attestations: true`.
- `examples/quickstart_cv.py` ships a runnable end-to-end CV tracking
  demo with vectorized `fx`/`hx` (previously only referenced in README).
- `.github/PULL_REQUEST_TEMPLATE.md` added.

**Rust quality:**
- `cargo fmt` + `cargo clippy --all-targets -- -D warnings` both clean.
  Pre-existing `non_local_definitions` lint from PyO3 0.20's macros is
  suppressed crate-wide with a documented `#![allow(...)]` (the upstream
  fix requires bumping to a newer PyO3, which is out of scope).

**Deferred (still open from the audit):**
- RTS smoother, square-root CKF, batched filtering with rayon, adaptive
  Q/R, multi-rate updates, additional standard models (CTRV/CTRA/Singer).
- `quaternion omega_cross` sign convention vs paper Eq. (3) — needs an
  actual hardware-trajectory cross-check, not just code review.
- Cross-backend benchmark vs Numba/JAX.
- pytest-benchmark history / CI regression gate.
- Apples-to-apples benchmark refactor (pointwise-vs-vectorized FilterPy
  ceiling/floor).
- `cargo audit` / `pip-audit` (not wired into the new CI yet).

---


Reviewed surface:
- `src/lib.rs` (~500 lines, PyO3 + nalgebra)
- `turbo_ckf/core.py`, `turbo_ckf/paper_ahrs.py`, `turbo_ckf/__init__.py`
- `turbo_ckf_tests/` (8 test files + 3 benchmarks)
- `.github/workflows/`, `Cargo.toml`, `pyproject.toml`, `PUBLISHING.md`, `README.md`

---

## 1. Bugs / correctness risks

### Rust backend (`src/lib.rs`)

- **`src/lib.rs:402-415`** — `stable_cholesky` silently adds jitter up to `1e-12 · 10⁷ = 1e-5` to the diagonal without recording it anywhere. The single biggest "wrong-results-silently" risk in the codebase. Surface jitter applied via a counter / flag in `snapshot()`.
- **`src/lib.rs:160-167`, `218-225`** — `try_inverse` falls back to SVD pseudo-inverse if `pzz` is singular, with no warning. A degenerate measurement covariance silently produces garbage K.
- **`src/lib.rs:157`, `215`** — Pxz subtracts `self.x · z_pred.T` rather than the sigma-point row mean of x. Exact in arithmetic when prior x equals the mean of sigma points; inconsistent with the `_ckf` recompute path under roundoff.
- **`src/lib.rs:173`** — Update uses `P - K·S·Kᵀ` directly with only `symmetrize`; no Joseph-form fallback. Over long runs P can drift non-PD, then `stable_cholesky` papers over it (see first item).
- **`src/lib.rs:311`** — Log-likelihood uses `lu().determinant()` of possibly indefinite S. Should use Cholesky log-det and bail when not PD.
- **`src/lib.rs:321`** — `likelihood = log_likelihood.exp()` underflows to 0 silently for large `dim_z`.
- **`src/lib.rs:394`** — Comment claims "scipy upper-triangular convention" but the code indexes `chol.l()` (lower-triangular) with `(j,k)` to effectively transpose. Code is correct; comment is misleading and fragile.
- **`src/lib.rs:35`** — `dt` is not validated. A NaN/negative dt propagates silently.
- **`src/lib.rs:286`** — `predict_kckf_linear` doesn't shape-check `F`; mismatch panics in nalgebra rather than raising `PyValueError`.
- **`src/lib.rs:200`** — `magnetic_reference_terms` derives the world reference from the current measurement, so a single bad sample skews `m_n, m_d`. Consistent with the paper, but should be documented as a calibration hazard.

### Python wrapper

- **`core.py:155-168`** — `np.asarray(..., dtype=float)` may return a *view* into a PyO3-owned buffer. If Rust reuses buffers across snapshots, user reads of `kf.P` could mutate underfoot. Force `.copy()`.
- **`core.py:115-122`** — `update(z=None)` is a wrapper-side no-op that copies `x → x_post`, `P → P_post`, but never touches the backend. `self.y`, `self.S`, `log_likelihood`, `mahalanobis` are left stale from the previous update — a silent landmine for NIS gating.
- **`core.py:150-171`** — Wrapper pushes only `x, P, Q, R` to the backend each step. If the user mutates other backend-side state (e.g., via `normalize_state_quaternion_backend`) the contract is asymmetric and undocumented.
- **`core.py:81-95`** — `predict_standard_model` accepts any string and forwards to Rust. Typos like `"constant_velocty"` surface as opaque Rust errors instead of a Python `ValueError` listing valid names.
- **`core.py:139-148`** — `update_paper_ahrs` doesn't validate that `sigma_acc2` / `sigma_mag2` are positive; zero or negative silently passes through and produces NaN K/S.
- **`core.py:202-211`** — `normalize_state_quaternion` mutates `self.x` but doesn't update `self.x_post` / `self.x_prior`, so introspection arrays drift out of sync until the next snapshot.
- **`core.py:307-313`** — `_coerce_args` wraps `list`/`np.ndarray` as a single positional arg. `predict(fx_args=[1,2,3])` becomes `fx(sigma, [1,2,3])`, not `fx(sigma, 1, 2, 3)` — FilterPy-style users will hit silent shape errors inside `fx`.
- **`paper_ahrs.py:29-37`** — `omega_cross` row 1 has `[wx, 0, wz, -wy]`. Sign convention should be cross-checked against the paper's Eq. (3) — JPL vs Hamilton confusion here would silently give 180°-wrong orientation.
- **`core.py:65-79`** — `predict(dt=...)` overrides are forwarded to backend but never written back to `self.dt`/`self.fx`. The next bare `predict()` reverts. Behaviour is reasonable; undocumented is the bug.

### Tests

- **`test_filterpy_parity.py:56`** — Asserts only state `x`, never `P`. Drops the harder check.
- **`test_math.py:71`** — `atol=1.1e-3` for `P` after 100 steps is suspiciously tight against the actual drift and has no documented rationale; any change crossing it produces a confusing failure.
- **`test_paper_alignment.py:42-88`** — "KCKF matches CKF" compares two internal implementations to each other; two impls agreeing proves nothing if both have the same bug.
- **`test_rust_backend.py:23-26`** — Only checks `_backend_name == "rust"`; doesn't assert numerical equivalence to a reference.
- **`test_api.py:82-86`** — `test_update_accepts_scalar_R` only checks symmetry of `S`; doesn't verify the scalar was actually broadcast to `(dim_z, dim_z)`.
- **`test_math.py:51`** — FilterPy gets a pointwise `fx` lambda while TurboCKF gets a matrix-form one. Not byte-identical; sigma-point evaluation order can diverge.
- **`test_vectorization.py:30-42`** — Tests that a user-raised `TypeError` propagates, not that TurboCKF itself rejects pointwise callbacks (which `README.md:25` claims it does).
- Missing edge-case coverage entirely: `dim_z > dim_x`, `dt=0`, near-singular `P`, non-PSD `Q`, NaN/Inf inputs, `dim_x ∈ {1, 50}`, mismatched-shape `z` in `update`.
- No seeded RNG anywhere; trajectories are deterministic ramps (`test_math.py:64`, `test_filterpy_parity.py:50`).
- No NIS / chi-squared innovation-consistency test (the standard correctness sanity check).

### Benchmarks

- **`benchmark.py:84-89`** and **`benchmark_paper.py:83-88`** — Single timed run per case; no repeats, warm-up, or median. The README's "3.56x / 4.37x / 1.67x" headline numbers cannot come from these scripts reproducibly.
- **`verify_before_after.py:52`**, **`benchmark.py:40`** — FilterPy is given a **pointwise** `fx`/`hx` while TurboCKF is given a **vectorized** one. FilterPy's CKF calls its callback `2·dim_x` times per step in pure Python — exactly the loop TurboCKF avoids. A meaningful fraction of the reported speedup is callback-vectorization, not Rust. Apples-to-apples comparisons need both a "FilterPy + vectorized-equivalent" ceiling and a "TurboCKF + pointwise-equivalent" floor.
- **`verify_before_after.py:142-157`** — Default `warmup=1` is too low; reports median of 7 without MAD/CI; no outlier rejection.
- **`benchmark_paper.py:49`** — Pre-allocates 250k 4×4 matrices in a Python list (~32 MB, hits L3) before timing; allocation cost is hidden in setup.
- **`README.md:73-77`** — Quotes `2.354375 s` to 6 decimals from a single-host, no-pinning run. False precision.
- No CPU pinning / governor / thermal note despite quoting M4 Max numbers. No GIL/threading benchmark. No memory benchmark.

### CI / release

- **`.github/workflows/release.yml:111,134`** — `publish-testpypi` and `publish-pypi` depend only on `build-wheels` / `build-sdist`; `ci.yml` is not a required predecessor. A broken `main` can be released.
- **`release.yml:34`** — Wheel matrix is only `ubuntu-latest`, `macos-latest`, `windows-latest` on default runners. Missing aarch64 Linux, ARM macOS (`macos-14`), musllinux, Windows ARM.
- No post-build job that `pip install`s the abi3 wheel and imports `turbo_ckf` on each of Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13.
- **`Cargo.toml:3` vs `pyproject.toml:7`** — Version `0.1.1` is hand-synced. Nothing in CI fails if they diverge or disagree with the pushed git tag.
- **`release.yml:148`** — Uses `pypa/gh-action-pypi-publish` without `attestations: true`. No `actions/attest-build-provenance`, no SBOM.
- **`.dockerignore` is 14 bytes** — just `PUBLISHING.md`. Does not exclude `target/`, `.venv-turbo-ckf/`, `.cargo/`, `.rustup/`, `__pycache__/`, or `*.so`. Docker builds will balloon and may leak `turbo_ckf/_rust.abi3.so` into images.
- **`.gitignore:36`** lists `PUBLISHING.md` while the file is committed. Inconsistent.
- **`README.md:97`** — Cites `arXiv:2602.12283 (2026)`. arXiv IDs are `YYMM.NNNNN`; `2602` is not a valid prefix. Likely a typo (verify against the actual paper) — users following the link get a dead reference.

---

## 2. Improvements (perf, refactors, DX)

### Rust performance

- **`src/lib.rs:483-503`** — `pyarray2_to_dmatrix` copies element-by-element. Replace with `DMatrix::from_iterator` over `arr.as_array().iter()` in column-major, or contiguity-check + `from_column_slice`. Hot path.
- **`src/lib.rs:506-516`** — `dmatrix_to_pyarray` builds `Vec<Vec<f64>>` then `PyArray2::from_vec2` — two heap allocations per row plus a final copy. Use `PyArray2::from_owned_array` (ndarray) or `unsafe { PyArray2::new }` + memcpy. Called every predict/update.
- **`src/lib.rs:95, 157, 215`** — `sigma.transpose() * &sigma` allocates a transpose. Use `sigma.tr_mul(&sigma)`.
- **`src/lib.rs:96, 158, 173, 287, 301`** — `symmetrize` allocates a transposed clone + sum. Rewrite in place: iterate `i<j` and average pairs.
- **`src/lib.rs:100-101, 181-182, 240-241, 288-289`** — `clone()` of `x`/`P` into prior/post every step. Make snapshot lazy (only on `snapshot()` call) behind a flag.
- **`src/lib.rs:390-398`** — `cubature_points` allocates a fresh `2n × n` matrix per call. Cache scratch buffers (`propagated`, `z_sigma`, `pxz`, `pzz`) on the struct.
- **`src/lib.rs:457-466`** — `row_mean` is a manual loop; `values.row_sum() * w` vectorizes.
- No `#[inline]` on `symmetrize`, `row_mean`, `cubature_points`.
- **`Cargo.toml:17`** — `rayon` is a dependency but is never used in `src/lib.rs`. Either remove or actually wire batched filtering (see Features).

### Python ergonomics

- **`core.py:259-279`** — Vectorized-callback error message is generic. The user who passes a pointwise `fx` gets "model output must have shape (4, 2), got (2,)" with no link to the README's callback contract. Embed the contract in the error.
- **`core.py:21-63`** — Add `__repr__`, `reset()`, `copy()`, `__getstate__` / `__setstate__` (pickling), `to_dict()` / `from_dict()`.
- **`__init__.py:1-5`** — Re-export `paper_ahrs` helpers from the package root. README treats AHRS as first-class but users currently must import from the submodule.
- **`core.py:65-148`** — No type hints on `predict`, `update`, `predict_standard_model`, `update_paper_ahrs`. Constructor is typed; methods are not — breaks IDE help where it matters most.
- **`core.py:81-113`** — Four near-identical `predict_*` methods with no shared overview of *when to use which*. Rename `predict_standard_model` → `predict_linear_closed_form` and `_ckf` variant → `predict_linear_sigma`, and add a single "choosing a predict path" docstring.

### Test methodology

- Add `hypothesis` property tests: `predict(dt=0)` is identity, `update(z)` with `R→∞` leaves state unchanged, `P` stays PSD across `predict→update`, scalar `R` broadcasts.
- Long-run numerical stability test (10k–100k steps) asserting `P` PSD (`eigvalsh > -eps`), symmetry (`||P − Pᵀ||`), bounded condition number. Nothing today runs >200 steps.
- NIS consistency test on simulated linear-Gaussian data: time-averaged NIS within chi-squared bounds.
- Canonical-trajectory convergence test (CV target, noisy position obs) asserting RMSE-vs-truth within an analytical bound — tests *correctness*, not just internal self-consistency.
- Bench scripts: enforce repeats + median + MAD, disable GC, randomize case order, fix a seed, log CPU model/governor/`RUSTFLAGS`/maturin profile alongside `python`/`platform`/`numpy`.
- Cross-backend benchmark (Numba, JAX) to honestly bound the Rust speedup. Without this the "Rust wins" framing is misleading.
- Use `pytest-benchmark --benchmark-autosave`, commit a baseline under `.benchmarks/`, and gate PRs with `--benchmark-compare-fail=mean:5%`. Or migrate to `asv`.
- Wire a CI job that consumes `verify_before_after.py`'s `compare_reports` output and fails on >10% regression / `parity.max_abs_state_diff > 1e-8`.

### CI / release / packaging

- Gate `publish-*` on a `test` job that installs the built wheel and imports it across 3.9–3.13 on each OS.
- Add a single CI step that diff-checks the version in `Cargo.toml`, `pyproject.toml`, and the pushed tag — fail if they don't agree.
- Add `cargo fmt --check` and `cargo clippy -- -D warnings` to `ci.yml`.
- Add `cargo audit`, `pip-audit`, and a `dependabot.yml`.
- Set `attestations: true` on `pypa/gh-action-pypi-publish`; add `actions/attest-build-provenance` and a CycloneDX SBOM step.
- Fix `.dockerignore` to exclude `target/`, `.venv-*`, `.cargo/`, `.rustup/`, `__pycache__/`, `*.so`.
- Ship `turbo_ckf/py.typed` + `.pyi` stubs for the public API.
- Modernize licensing per PEP 639: `license = "MIT OR Apache-2.0"` + `license-files = ["LICENSE-MIT", "LICENSE-APACHE"]` in `pyproject.toml`.
- Add Python 3.13 to the classifier list in `pyproject.toml:23-26` (abi3 already supports it).
- Add `concurrency` keyed on tag in `release.yml:23-25` so a double manual dispatch doesn't queue two publish attempts.
- `setup_env.sh` `curl | sh`s rustup unconditionally — gate behind "is `cargo` already installed?" and document the Windows path (or add `Makefile`/`justfile` targets so contributors don't read shell to find the test command).
- Add `CONTRIBUTING.md`, issue/PR templates under `.github/`, `pre-commit` config (`ruff`, `rustfmt`, `clippy`).
- Add `CHANGELOG.md` (Keep-a-Changelog format). PyPI users currently have no record of what changed in 0.1.1.
- README polish: lead `## Install` with `pip install turbo-ckf`; add CI/PyPI/license/Python-versions badges to `README.md:1`; add a copy-pasteable example that actually defines `hx_vectorized` / `fx_vectorized` (currently referenced but never shown).
- Add an `examples/` directory; today the only runnable code lives in `turbo_ckf_tests/`.
- Stand up an API reference (`pdoc` is one command and ships to GitHub Pages).
- Verify the arXiv ID at `README.md:97` and fix the year if needed.

---

## 3. Features worth adding

### Filter algorithms (highest-leverage)

1. **RTS smoother** (`rts_smooth(xs, Ps, Fs, Qs)`) — the standard companion to any KF; trivial given existing primitives. Currently absent.
2. **Square-root CKF (SR-CKF)** — propagate the Cholesky factor `S` instead of `P`. This removes the `stable_cholesky` silent-jitter problem at the source (see bug list). Most-requested CKF variant in the literature.
3. **Batched / parallel filtering over independent state vectors** — `batch_predict_update(states, zs)`. This is where the currently-unused `rayon` dependency would actually pay off (Monte-Carlo, particle banks, multi-target tracking).
4. **Adaptive Q/R** — Sage-Husa / IAE / RAE. Easy add over a stored innovation window.
5. **Multi-rate / partial measurement update** — `update_subset(z, idx, hx)` for asynchronous sensors.
6. **Additional standard models** — currently only constant-velocity / constant-acceleration. Add CTRV/CTRA (tracking), Singer, Wiener-process acceleration, jerk.

### API additions

7. **Per-step diagnostics return** — `update()` currently returns `()`. Optionally return `{y, S, NIS, log_likelihood}` so users don't `snapshot()` every step.
8. **NIS / chi-square gating helper** — `kf.nis()` and `kf.gate(z, threshold)`. The #1 thing users add by hand.
9. **`batch_filter(zs, dts=None)`** — Rust-side batch loop. Unlocks another order of magnitude for the standard-model path; FilterPy has the equivalent.
10. **`residual_fn` / `subtract` hook** — needed for manifold innovations (quaternions). The AHRS path presumably handles this internally; the general `update(z)` does not expose it.
11. **State serialization** — `to_dict()` / `from_dict()` over the snapshot. Pickling the Rust object is brittle; a dict round-trip is portable.
12. **Reset semantics** — `reset()`, `reset_likelihood()`, and a "has any update happened?" query for warm-start logic.
13. **`TurboCKF.from_filterpy(ckf)` classmethod** + auto-vectorizing wrapper for pointwise `fx`/`hx` (via `np.apply_along_axis`). Meaningfully lowers migration cost — the README explicitly positions this as a FilterPy replacement.

### Tooling / observability

14. **Diagnostics surface for `stable_cholesky` jitter** — counter on the struct, exposed via `snapshot()`. The single biggest "fix it before more bugs hide behind it" item.
15. **Docs site** (pdoc / mkdocs) with the public API + a "porting from FilterPy" guide.
16. **`examples/`** — AHRS quick-start (with real IMU data), CV tracking, side-by-side FilterPy comparison.

---

## Top-priority shortlist

If only six items get done:

1. Surface `stable_cholesky` jitter via the snapshot (silent-correctness bug, `src/lib.rs:402-415`).
2. Fix the two Python↔Rust copy hot paths (`src/lib.rs:483, 506`).
3. Make the README benchmark numbers honest: vectorized-vs-pointwise apples-to-apples plus warmup/repeats (`verify_before_after.py`, `benchmark.py`, `benchmark_paper.py`, `README.md:60-93`).
4. Gate release on tests + version sync, add wheel-import smoke job for 3.9–3.13 (`.github/workflows/release.yml`).
5. Add `py.typed` + `.pyi` stubs + `CHANGELOG.md`, fix `.dockerignore` and PEP 639 license metadata (`pyproject.toml`, `.dockerignore`).
6. Add RTS smoother + NIS gating + per-step diagnostics return — three high-leverage features users will reach for first.

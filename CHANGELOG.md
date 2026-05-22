# Changelog

All notable changes to `turbo-ckf` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-05-21

### Added
- `TurboCKF.enable_adaptive_noise(window=30, mode="R", alpha=0.3,
  diagonal_floor=1e-12)` — opt-in Sage-Husa adaptive estimation of `R`
  (canonical, innovation-based) and/or `Q` (heuristic, state-correction
  outer-product with a documented stability caveat). After a `window`-step
  warm-up, each successful `update(z=...)` folds the new innovation into a
  running exponentially-weighted estimate and writes it back to `self.R`
  (and/or `self.Q`) before the next step. Tracks Q/R drift over long runs
  without the user having to retune by hand.
- `TurboCKF.disable_adaptive_noise()` and the read-only
  `TurboCKF.adaptive_noise_estimator` property for introspection /
  diagnostics. The estimator state (running estimates, sample count,
  configured `window`/`mode`/`alpha`) is carried through
  `TurboCKF.copy()` and `TurboCKF.to_dict()` / `from_dict()`; `reset()`
  drops it.
- New private estimator class `turbo_ckf.core._AdaptiveNoiseEstimator`
  containing the math (kept in Python so `copy()` / `to_dict()` round-trip
  state for free and the formulation stays inspectable).
- `turbo_ckf_tests/test_adaptive_noise.py`: 11 tests covering
  (1) default-OFF guarantee — no behavioral change vs the 0.6.0 surface,
  (2) `R` convergence to ~21% relative error vs a true `R = 4 × initial_R`
  in 500 steps with adaptive enabled (gate set at 25%),
  (3) time-averaged NIS pulled toward chi-squared mean for `dim_z`,
  (4) 10k-step PSD/symmetry invariants with adaptive `R` running,
  (5) `mode="both"` stability with conservative `alpha`,
  (6) `copy()` / `to_dict()` / `from_dict()` carry estimator state,
  (7) input validation (bad `window`, bad `alpha`, bad `mode`),
  (8) direct unit tests of the warm-up gate and diagonal-floor clamp.

### Notes
- Default is OFF: callers that don't invoke `enable_adaptive_noise()` see
  identical behaviour to v0.6.0; all 80 prior tests stay green with zero
  modification.
- Scope deliberately kept to the per-step `update(z=...)` path. The
  `update_paper_ahrs` path overwrites `R` from `sigma_acc2`/`sigma_mag2`
  each call (an adaptive write would be discarded); the estimator skips
  there. The `batch_filter` / `batch_parallel_step` static methods carry
  their own Rust-side `R` and are not affected by the per-instance
  adaptive state — adaptive batch paths would require moving the
  estimator into Rust, deferred to a later session.
- Q-channel is a heuristic (`K · y · y^T · K^T`) and not unbiased under
  arbitrary dynamics. The docstring flags this and recommends keeping
  `alpha` small + verifying NEES on a held-out trajectory; the diagonal
  floor keeps `Q` PD even when the contribution drifts.

## [0.6.0] - 2026-05-21

### Added
- `TurboCKF.batch_parallel_step(xs, Ps, zs, F, H, Q=None, R=None)` — parallel
  linear predict+update over a bank of `M` independent Kalman filters that
  share `(F, H, Q, R)`. Each filter has its own `(x_i, P_i)` and a single
  observation `z_i`; the `M` filter steps run in parallel via `rayon` with
  the GIL released. Distinct from `batch_filter` ("one filter, many
  observations") — this is the "many filters, one observation each"
  pattern used in Monte-Carlo banks, particle filters, and multi-target
  tracking. Activates the previously-unused `rayon` dependency.
- Returns `(xs_new, Ps_new, log_likelihoods, status)`. `status[i]` reports
  per-filter health: `0` ok, `1` singular innovation (pseudo-inverse used),
  `2` no inverse at all (measurement update skipped, ll = `-inf`). One
  failing filter does not abort the bank — Monte-Carlo callers can mask
  on `status` and continue.
- New Rust free function `turbo_ckf._rust.batch_parallel_step` backing the
  wrapper, with a non-`PyResult` `invert_innovation_noraise` helper so
  per-filter failure can be reported as a status code instead of an
  exception that would abort the whole bank.
- `turbo_ckf_tests/test_batch_parallel_step.py`: 6 tests covering ~1e-10
  state + covariance + log-likelihood parity vs a sequential per-step
  reference loop on `M=100`; shape/dtype/defaults/rejection-of-bad-shapes
  coverage; per-filter error handling (singular `S` in one filter does
  not break the others); and a perf gate enforcing `>= 4x` speedup over
  the sequential Rust per-step loop on `M=10,000`.
- Re-exported as `turbo_ckf.batch_parallel_step`.

### Performance
- Bank of `M=10,000` 4-state filters with a 2D linear measurement: the
  sequential Rust per-step loop takes ~125 ms; one parallel step takes
  ~1.8 ms — **~70x speedup** on M-class Apple silicon (8 performance
  cores plus per-filter parallelism amortizing FFI / Python overhead).
  The "at least 4x" target is well exceeded.

### Notes
- Scope deliberately kept to the linear path (`predict_linear_model` +
  linear measurement update). Nonlinear `predict_custom` / cubature-
  update via Python `fx` / `hx` callbacks would serialize on the GIL
  during every callback — defer to a later session with a pure-Rust
  callback contract.

## [0.5.0] - 2026-05-21

### Added
- `TurboSRCKF` — Square-Root Cubature Kalman Filter, a new estimator class
  alongside `TurboCKF`. Propagates the lower-triangular Cholesky factor of
  P directly instead of P itself. Predict step uses a single QR of the
  stacked weighted sigma-point deltas with `chol(Q)`; update uses one QR
  for the innovation factor plus per-measurement-dim rank-1 Cholesky
  downdates for the posterior factor. The filter loop never calls
  `stable_cholesky` on P, so the silent diagonal-jitter accumulation that
  `TurboCKF` reports via `jitter_count` is structurally impossible in
  steady state — the new `downdate_fallback_count` diagnostic surfaces
  the rare extreme-conditioning case where a rank-1 downdate would break
  PD and we have to rebuild P_post explicitly. Mirrors the
  `predict_custom` + `update` surface from `TurboCKF`; standard-model
  predict paths and the paper AHRS update stay on `TurboCKF`. Re-exported
  as `turbo_ckf.TurboSRCKF`.
- New Rust class `turbo_ckf._rust.SquareRootCubatureKalmanFilter` backing
  the wrapper. Exposes `set_state`, `predict_custom`, `update`,
  `clear_update_diagnostics`, `reset_jitter_counters`, and `snapshot`.
  Snapshot includes `chol_P`, `chol_Q`, `chol_R`, `S_innov`, the derived
  `P` / `Q` / `R` / `S`, and the full diagnostics surface
  (`last_jitter`, `max_jitter`, `jitter_count`,
  `singular_innovation_count`, `downdate_fallback_count`).
- `turbo_ckf_tests/test_sr_ckf.py`: 8 tests covering (1) ~1e-8 parity vs
  `TurboCKF` on a 60-step linear-Gaussian CV trajectory and a 200-step
  pure-predict chain; (2) ill-conditioned trajectory (rank-deficient
  initial P with eigenvalue ratio ~1e18 + tiny R) where `TurboCKF` must
  add jitter via `stable_cholesky` while `TurboSRCKF.jitter_count` stays
  at zero; (3) snapshot factor-consistency (`chol_P · chol_P^T == P`),
  skipped-update diagnostic clearing, constructor validation, reset, and
  chi-square gating.

### Notes
- Why a separate class vs a backend flag on `TurboCKF`: the struct state
  is genuinely different (a factor, not a covariance), so a flag would
  have meant two code paths through every existing predict variant +
  update + AHRS method, putting all 62 prior tests at risk. The separate
  class keeps the opt-in clean and preserves the existing surface
  unchanged.

## [0.4.0] - 2026-05-21

### Added
- `TurboCKF.batch_filter(x0, P0, zs, F, H, Q=None, R=None)` linear Kalman
  batch filter, backed by a new Rust `turbo_ckf._rust.batch_filter_linear`
  free function. Single Rust-side loop with Joseph-form posterior, runs
  the full predict/update sequence without crossing Python on each step.
  Returns `(xs, Ps, log_likelihoods)` directly consumable by
  `rts_smooth` — no reshape between forward and backward passes.
  Per-step `F`/`H`/`Q`/`R` (leading dimension `N`) or constants are both
  accepted; constants are broadcast in Python. Re-exported as
  `turbo_ckf.batch_filter`.
- `turbo_ckf_tests/test_batch_filter.py`: 8 tests including a tight
  numerical-parity check against a sequential `predict_linear_model` +
  `update` loop (~1e-9 agreement on state, covariance, and log-
  likelihood), a `batch_filter → rts_smooth` composition test on a
  CV trajectory, constant-vs-tiled broadcast equivalence, defaults
  (Q=0, R=I), per-step F handling, log-likelihood sanity, and shape
  rejections.

### Performance
- 10,000-step linear CV filter: 118 ms sequential (Python ↔ Rust per
  step) → 5.3 ms via `batch_filter` — **~22x speedup** on M-class
  Apple silicon, comfortably clearing the "order of magnitude" target.

## [0.3.0] - 2026-05-21

### Added
- `TurboCKF.rts_smooth(xs, Ps, Fs, Qs)` Rauch-Tung-Striebel fixed-interval
  smoother, backed by a new Rust `turbo_ckf._rust.rts_smooth` free function.
  Per-step `Fs`/`Qs` accept either length `N` (FilterPy-compatible — last
  entry unused) or length `N-1`, so non-constant transitions are handled
  natively without callbacks. Re-exported as `turbo_ckf.rts_smooth`.
- `turbo_ckf_tests/test_rts_smoother.py`: forward-filter-then-smooth on a
  canonical linear-Gaussian constant-velocity trajectory, asserting at
  least a 30% drop in position RMSE versus the forward-only estimate, plus
  endpoint-equality, single-step identity, length-`N` vs length-`N-1`
  agreement, non-constant `F` handling, shape-validation rejections, and
  the `TurboCKF.rts_smooth` static-method form.

## [0.2.0] - 2026-05-21

### Added
- `TurboCKF.nis`, `last_jitter`, `max_jitter`, `jitter_count`, and
  `singular_innovation_count` attributes, all pulled from the backend
  snapshot each step. NIS is the squared Mahalanobis distance; the jitter
  fields surface previously silent Cholesky conditioning.
- `TurboCKF.gate(threshold)` for chi-square gating against the most recent
  innovation.
- `TurboCKF.reset()`, `TurboCKF.copy()`, `__deepcopy__`, `__repr__`.
- `TurboCKF.to_dict()` / `TurboCKF.from_dict()` for portable serialization
  (callbacks are re-supplied on restore).
- `paper_ahrs` helpers (`normalize_quaternion`, `transition_matrix_from_gyro`,
  `process_noise_from_quaternion`, `measurement_noise`,
  `magnetic_reference_terms`, `observation_model`) re-exported from the
  package root.
- New `turbo_ckf_tests/test_correctness_invariants.py` (16 tests): 10k-step
  P symmetry / PSD, NIS chi-squared sanity bounds, jitter counter, `dt=0`
  identity, `dim_z > dim_x`, scalar `R` broadcast via math, mismatched-z
  rejection, `update(z=None)` diagnostic clearing, `reset` / `copy` /
  `to_dict` round-trips, gate accept/reject, input-validation rejections.
- `examples/quickstart_cv.py` with defined vectorized `fx`/`hx`.
- PEP 561 `turbo_ckf/py.typed` marker.
- `CONTRIBUTING.md`, `Makefile`, `.github/dependabot.yml`, and
  `.github/PULL_REQUEST_TEMPLATE.md`.
- README install badges, `pip install turbo-ckf` quick-start.

### Changed
- Python wrapper public methods now carry type hints.
- `predict_standard_model[_ckf]` validates `model_type` against a whitelist
  surfaced by the backend; typos raise `ValueError` with the valid names.
- `update_paper_ahrs` validates `sigma_acc2` and `sigma_mag2` are finite
  and positive in the Python wrapper before reaching Rust.
- Constructor validates `dt` is finite.
- `_coerce_args` now unpacks `list`-typed callback args (was silently
  wrapping them as a single positional arg).
- Better error messages when `fx` / `hx` return the wrong shape; point at
  the vectorized-callback contract.
- Rust hot paths: `pyarray2_to_dmatrix` uses `from_row_iterator`,
  `dmatrix_to_pyarray` builds a single `Vec<f64>` + `ndarray::Array2`
  instead of `Vec<Vec<f64>>`, `tr_mul` replaces explicit
  transpose-then-multiply, `symmetrize_in_place` avoids transpose clones,
  hot helpers marked `#[inline]`.
- Modernized license metadata in `pyproject.toml` to PEP 639 form.
- Tightened `.dockerignore` to exclude `target/`, virtualenvs, caches,
  compiled extensions, and other build artifacts.
- CI split into `rust-lint` (`cargo fmt --check`, `cargo clippy -- -D
  warnings`) and `test` (Python 3.9-3.13 matrix).

### Fixed
- `update(z=None)` previously left stale `y`, `S`, `SI`, `K`,
  `log_likelihood`, `mahalanobis`, `nis` from the prior real update. Now
  cleared via a new backend `clear_update_diagnostics` entry point so
  downstream NIS gates can't misread the previous innovation.
- `stable_cholesky` jitter was added silently up to `1e-5`. The backend
  now reports `last_jitter`, `max_jitter`, and `jitter_count` so callers
  can detect when P is being conditioned.
- Log-likelihood is now computed via the Cholesky factor of S
  (`2 * sum(log(diag(L)))`), failing cleanly to `-inf` instead of
  drifting through an indefinite LU determinant.
- Singular innovation-covariance fallbacks are counted
  (`singular_innovation_count`) rather than silently swapping in a
  pseudo-inverse.
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` pass on
  the backend (one pre-existing PyO3 0.20 macro lint is suppressed at
  the crate level with a documented `#![allow(non_local_definitions)]`).

### Security
- Release workflow requires version agreement between `Cargo.toml`,
  `pyproject.toml`, and the pushed tag before publishing.
- PyPI publish step requests build attestations.

## [0.1.1] - 2026-02-21

### Added
- Local benchmark results and paper reference in the README.
- Manual release dispatch path that accepts a tag input.

### Fixed
- Release workflow shell selection and Python interpreter resolution.
- `maturin develop` now runs inside a freshly created virtualenv on CI.

## [0.1.0] - 2026-02-18

### Added
- Initial public release of `turbo-ckf`, a Rust-backed Cubature Kalman Filter
  with a FilterPy-style Python wrapper.
- Standard-model predict paths (constant velocity, constant acceleration).
- KCKF-style AHRS update path from Yamagishi and Jing (arXiv:2602.12283).
- Parity tests against FilterPy and benchmark scripts.

[Unreleased]: https://github.com/mokhld/turbo-ckf/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/mokhld/turbo-ckf/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/mokhld/turbo-ckf/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/mokhld/turbo-ckf/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/mokhld/turbo-ckf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mokhld/turbo-ckf/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mokhld/turbo-ckf/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/mokhld/turbo-ckf/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mokhld/turbo-ckf/releases/tag/v0.1.0

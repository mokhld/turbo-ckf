# Feature backlog

Agent-ready feature specs for turbo-ckf. To start one, point an agent at this file:

> Implement the next open feature in docs/FEATURES.md.

or name one:

> Implement B4 from docs/FEATURES.md.

Each item is self-contained: the user need, the API to build, where the code goes, and
the acceptance criteria that define "done". Priorities and evidence come from the review
in `docs/REVIEW.md` (Part B uses the same IDs).

## Protocol for the implementing agent

1. Read `CLAUDE.md` (package map, build and test commands, gotchas), then the item below.
   If the person named no item, take the first row in the status table whose status is
   `open` and whose dependencies are all `done`.
2. Set the item's status to `in progress` in the table below before writing code.
3. Work on a branch named `feat/<id>-<slug>` off `main`. Commit or push only if the
   person asked you to.
4. Build to the spec. If the spec is wrong or ambiguous in a way that changes the public
   API, stop and ask the person instead of guessing. Smaller calls are yours; list them
   in your report.
5. Verify with the commands in `CLAUDE.md` "Build, test, run": rebuild the extension,
   run the full pytest suite (all pass, Python line coverage stays at 100%), and run
   `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings`. Then run
   the item's own acceptance checks and record the numbers.
6. Update documentation:
   - this file: status `done`, the date, a one-line outcome under the item, and any
     follow-up work as a new item at the bottom of the table;
   - `CHANGELOG.md` under `## [Unreleased]` (Keep a Changelog style);
   - `README.md` for any public API change;
   - `CLAUDE.md` if the code map or conventions changed;
   - `docs/REVIEW.md` Part B status for the same ID.
7. Report: what changed (files and functions), tests added, acceptance evidence with
   numbers, and judgment calls.

Standards that apply to every item: surgical changes that match the surrounding style;
tests for all new behavior, including invalid input; no new runtime dependencies beyond
`numpy` without asking; prose without em or en dashes; `turbo_ckf/_rust.abi3.so` is a
build artifact and must be rebuilt after any `src/lib.rs` change.

## Status

| ID | Feature | Effort | Status | Depends on |
| -- | ------- | ------ | ------ | ---------- |
| B1 | Rust-owned filter state (lazy attribute sync) | medium | done (PR #25) | none |
| B2 | Wheels for aarch64 Linux and Intel macOS | small | open | none |
| B3 | One-call AHRS pipeline in Rust | medium | open | A2 fix |
| B4 | Missing data and diagnostics in `batch_filter` | small | open | A5 fix |
| B5 | Cubature RTS smoother for nonlinear runs | medium | open | B1 |
| B6 | FilterPy migration helpers | small | open | A6 fix |
| B7 | `TurboSRCKF` feature parity | medium | open | B1 |
| B8 | Upgrade PyO3, rust-numpy, nalgebra | medium | open | none |
| B9 | Rust unit tests and a `cargo test` CI job | small | open | none |
| B10 | Nonlinear parallel filter bank | large | open | B1 |
| B11 | Filter consistency checks (NEES/NIS) | small | open | none |
| B12 | `reset_jitter_counters()` on `TurboCKF` | small | open | none |

The A-numbered dependencies are bug fixes from `docs/REVIEW.md` Part A. As of 2026-09-24,
A1-A8 are all fixed (PRs #20-#25), so every item above is unblocked except where it
depends on another open B item.

## B1. Rust-owned filter state (lazy attribute sync)

Done 2026-09-24 in PR #25. Outcome: 12.6 -> 4.1 us/step for `predict_standard_model` +
`update` (4.7 when reading x and P each step), callback path 15.7 -> 6.8, `TurboSRCKF`
18.9 -> 6.6, `run()` over 10k steps 182 -> 51 ms. `run()` stayed a Python loop that calls
the backend directly; moving it into Rust would save at most ~18%. Spec kept below as a
record of the requirements.

- Need: per-step speed, the package's core promise. Before this work, every wrapper call
  pushed x/P/Q/R to Rust and pulled a ~22-array snapshot back, about 71% of a 12.3 us
  step for a 4-state/2-measurement model. The same round trip made `TurboSRCKF`
  re-factor P every step (REVIEW A3).
- Requirements: all public attributes readable with unchanged names, shapes and values;
  x/P/Q/R setters keep validating; in-place edits of returned arrays either take effect
  on the next call or fail loudly, never silently; reading attributes never re-factors
  or changes results; `TurboSRCKF` keeps `chol_P` in Rust; copy/from_dict keep counters;
  both classes pickle.
- Acceptance: wrapper `predict_standard_model` + `update` loop at most 6 us/step on the
  reference machine when no attributes are read; SR ill-conditioned test asserts
  `max_jitter == 0` after `reset_jitter_counters()`.

## B2. Wheels for aarch64 Linux and Intel macOS

- Need: `pip install turbo-ckf` on a Raspberry Pi, Jetson or AWS Graviton host (the
  paper's second benchmark platform is a Raspberry Pi 4) and on Intel Macs. v0.8.0 ships
  wheels only for manylinux x86_64, macOS arm64 and Windows x64, so those hosts build
  from the sdist and need a Rust toolchain.
- Build: in `.github/workflows/release.yml` `build-wheels`, add
  - Linux aarch64: `PyO3/maturin-action@v1` with `target: aarch64` and
    `manylinux: auto` (cross-compiles in the manylinux container), or a native
    `ubuntu-24.04-arm` runner;
  - macOS x86_64: `target: x86_64` on `macos-latest` (cross-compile).
  The wheels are abi3 (`abi3-py39`), so one wheel per platform covers Python 3.9+.
  Add a smoke-test job per new wheel: `ubuntu-24.04-arm` for aarch64, and for x86_64
  macOS an Intel runner if GitHub still offers one (check the current runner list),
  otherwise an x86_64 Python under Rosetta (`arch -x86_64`) on an arm64 runner. Install
  the wheel into a fresh venv without the source tree on the path, then import
  `turbo_ckf` and run one `predict()` + `update()`.
  Update the README "Install" section to list the platforms.
- Acceptance: a `workflow_dispatch` run with `repository=testpypi` produces 5 wheels plus
  the sdist, and every smoke test passes. This can't run locally: prepare the change,
  then ask the person to trigger the workflow and share the run link before you mark it
  done.
- Files: `.github/workflows/release.yml`, `README.md`.

## B3. One-call AHRS pipeline in Rust

- Need: the paper's selling point is speed, but each AHRS step today is four wrapper
  calls (`process_noise_from_quaternion` in NumPy, `predict_linear_model`,
  `update_paper_ahrs`, quaternion normalization). Per the README, the wrapper cuts the
  KCKF-vs-CKF gain from 1.67x in the backend to 1.03x.
- API: `turbo_ckf.ahrs_filter(q0, P0, gyro, acc, mag, dts, gyro_variance, sigma_acc2,
  sigma_mag2, mag_gate=None)`, also exported from the package root. Returns an
  `AhrsRun` named tuple: `qs (N, 4)`, `Ps (N, 4, 4)`, `nis (N,)`, `used_mag (N,) bool`.
  - `gyro (N, 3)` in rad/s, `acc (N, 3)` and `mag (N, 3)` in any units (each row is
    normalized, matching `update_paper_ahrs` after the A2 fix); `dts` is a scalar or
    `(N,)`.
  - Step k: `gyro[k]` and `dts[k]` drive the predict from k-1 to k, the same indexing as
    `batch_filter`'s `Fs[k]`. F from Eq. (3), Q from Eqs. (4)-(5) at the current
    quaternion, KCKF linear predict, the Eq. (9)/(12)-(14) update, then renormalize the
    quaternion.
  - `mag=None`, or an all-NaN `mag[k]` row: accelerometer-only update for that step
    (the first three rows of the observation model, 3x3 R from `sigma_acc2`).
  - `mag_gate`: optional chi-square threshold on the 6-D NIS. When exceeded, redo the
    step as accelerometer-only and set `used_mag[k] = False` (a magnetic-disturbance
    guard).
  - Validate shapes and finiteness up front, before any step runs (an all-NaN `mag` row
    is the only allowed NaN).
- Build: a new `#[pyfunction]` in `src/lib.rs` that reuses `paper_observation_model`,
  `magnetic_reference_terms` and the KCKF predict math, with no Python calls in the
  loop. Put the Python wrapper and `AhrsRun` next to the helpers in
  `turbo_ckf/paper_ahrs.py` (or `core.py` if you need FilterRun-style plumbing). Add a
  README example.
- Acceptance: on a 1000-step synthetic trajectory (known rotation rates, simulated
  noisy acc/mag), results match the equivalent Python loop of existing calls to 1e-10;
  at least 10x faster than that loop (report numbers); accelerometer-only mode recovers
  roll and pitch to under 1 degree on synthetic data; the mag gate rejects an injected
  magnetic disturbance; tests cover invalid shapes and NaN rows.

## B4. Missing data and diagnostics in `batch_filter`

- Need: `batch_filter` is about 22x faster than a per-step loop, but real sensor logs
  have dropouts, and after the A5 fix it raises on any non-finite `zs`. Users with real
  logs are pushed back to the slower `run()`.
- API (backward compatible): `TurboCKF.batch_filter(x0, P0, zs, F, H, Q=None, R=None,
  nan_means_missing=False, full_output=False)`.
  - `nan_means_missing=True`: an all-NaN row of `zs` means "predict only" for that step
    (same meaning as in `run()`); rows mixing NaN and finite values still raise.
  - `full_output=True`: return a `FilterRun` (xs, Ps, x_priors, P_priors,
    log_likelihoods, nis, missing), the same type `run()` returns. Log-likelihood and
    NIS are NaN at missing steps. Otherwise return the current `(xs, Ps, lls)` tuple.
- Build: extend `batch_filter_linear` in `src/lib.rs` to accept a missing mask and to
  emit priors and NIS (always compute them; they are cheap). Keep validation of the
  non-missing inputs from the A5 fix.
- Acceptance: with dropouts, results match `run()` driven by linear `fx`/`hx` callbacks
  to 1e-9 (states, covariances, priors, log-likelihoods, NIS); `rts_smooth` over the
  output works; `full_output=False` output is bit-identical to today's; a 10,000-step
  run stays at least 15x faster than `run()` (report numbers).

## B5. Cubature RTS smoother for nonlinear runs

- Need: `rts_smooth` needs linear `Fs`, so users of the nonlinear CKF (the package's
  namesake) cannot smooth at all. Offline trajectory reconstruction is a routine need.
- API: `FilterRun` gains `cross_covs (N, dim_x, dim_x)`: entry k is the cross-covariance
  between the posterior at k-1 and the prior at k (entry 0 is relative to the initial
  state). Add `turbo_ckf.ckf_rts_smooth(run) -> (xs_smooth, Ps_smooth)`, also exported
  from the package root, and `run(..., smooth=True)`, which adds `xs_smooth` and
  `Ps_smooth` to the result.
- Math: in the predict step, with sigma points X_i of the posterior at k-1 and
  propagated points X*_i, `C_k = (1/2n) sum_i (X_i - x_{k-1}) (X*_i - x_k^-)^T`.
  Backward pass for k = N-2 ... 0: `G = C_{k+1} (P_{k+1}^-)^{-1}`,
  `x_s[k] = x[k] + G (x_s[k+1] - x_{k+1}^-)`,
  `P_s[k] = P[k] + G (P_s[k+1] - P_{k+1}^-) G^T`. Use the existing
  `invert_innovation` fallback path for the inverse.
- Build: compute `C_k` in Rust `predict_custom` (the sigma points are already there) for
  both filter classes. Store it in backend state and expose it through the B1 getters.
  Do the backward pass in Rust.
- Acceptance: on a linear model the result equals `batch_filter` + `rts_smooth` to 1e-9;
  on a nonlinear benchmark (for example a pendulum or a coordinated turn with range and
  bearing measurements) it matches a NumPy reference implementation to 1e-10 and cuts
  RMSE versus the filtered estimate by at least 20% (report numbers); it also works on a
  `TurboSRCKF` run.

## B6. FilterPy migration helpers

- Need: FilterPy migrants are the main audience. The A6 fix adds
  `predict_standard_model(..., layout="blocked" | "interleaved")`. What is still missing
  is a matching Q helper, a per-call `dt`, and a migration guide.
- API:
  - `turbo_ckf.q_discrete_white_noise(dim, dt, var, block_size=1, layout="interleaved")`,
    equal to `filterpy.common.Q_discrete_white_noise` (`layout="interleaved"` is
    FilterPy's `order_by_dim=True`, its default; `layout="blocked"` is
    `order_by_dim=False` and matches the standard models' default layout). Support
    `dim` 2, 3 and 4 like FilterPy.
  - `dt=None` keyword on `predict_standard_model` and `predict_standard_model_ckf`,
    overriding `self.dt` for that call only, like `predict(dt=...)`.
  - README section "Migrating from FilterPy" that maps `KalmanFilter`,
    `UnscentedKalmanFilter`, `batch_filter`, `rts_smoother`, `Q_discrete_white_noise`
    and `order_by_dim` to turbo-ckf equivalents. It should call out the vectorized
    callback contract and the `Fs` indexing convention.
- Acceptance: `q_discrete_white_noise` equals FilterPy's output to 1e-15 for every
  `dim`, `block_size` 1-3 and both layouts; the `dt` override matches a filter
  constructed with that `dt`; the README examples run as written (add a test that
  executes them, or copy them into a test).

## B7. `TurboSRCKF` feature parity

- Need: users who pick the square-root filter for conditioning lose the fast linear
  predict, the AHRS update, and adaptive noise.
- API: on `TurboSRCKF`, add `predict_linear_model(F)`,
  `predict_standard_model(model, layout=...)` (QR of `[F chol_P, chol_Q]`, no cubature
  points needed), `update_paper_ahrs(...)` (same normalization as `TurboCKF`), and
  `enable_adaptive_noise(...)` (write back a PSD-projected R through its Cholesky
  factor).
- Acceptance: each method matches `TurboCKF` to 1e-8 on well-conditioned problems. On
  the ill-conditioned scenario from `test_sr_ckf.py`, no in-loop jitter
  (`max_jitter == 0` after `reset_jitter_counters()`).

## B8. Upgrade PyO3, rust-numpy, nalgebra

- Need: PyO3 0.20 and rust-numpy 0.20 are from 2023. Official NumPy 2 support in
  rust-numpy arrived in 0.21, and newer PyO3 drops the
  `#![allow(non_local_definitions)]` workaround and supports newer Pythons natively.
- Build: move to the current PyO3 (Bound API) and a matching rust-numpy, and update
  nalgebra. Keep `abi3-py39`. Remove the crate-level allow.
- Acceptance: all tests pass; no per-step slowdown beyond 5% on the B1 benchmark; the
  extension builds from source under Python 3.9 and 3.13 (and 3.14 if available);
  clippy is clean without the allow.

## B9. Rust unit tests and a `cargo test` CI job

- Need: the Rust math (`cholesky_downdate`, `qr_to_lower_factor`, `transition_matrix`,
  `stable_cholesky`, `invert_innovation`) is only tested through Python. A plain
  `cargo test` does not link today, because `pyo3/extension-module` is enabled in
  `Cargo.toml`.
- Build: enable `extension-module` only through maturin (`pyproject.toml` already passes
  `features = ["pyo3/extension-module"]`) and drop it from the `Cargo.toml` pyo3 feature
  list. Check that wheels still build: `maturin build --release`, then inspect the wheel.
  Add `#[cfg(test)]` unit tests for the helpers above and a `cargo test` step to the
  `rust-lint` job in `ci.yml`.
- Acceptance: `cargo test` passes locally and in CI; the release wheel still imports.

## B10. Nonlinear parallel filter bank

- Need: `batch_parallel_step` is linear-only. Monte-Carlo and multi-target users with
  nonlinear models loop in Python.
- API: `TurboCKF.bank_step(xs, Ps, zs, fx, hx, Q, R, dt, fx_args=(), hx_args=())`
  advances M filters by one predict and update. `fx` is called once with all
  `M * 2n` sigma points stacked (`(M*2n, n)`) and `hx` once with all propagated points,
  so there are two Python crossings per bank step instead of 2M. Return
  `(xs_new, Ps_new, lls, status)` with the status codes of `batch_parallel_step`.
- Build: sigma-point generation and moment math in Rust with rayon, GIL released outside
  the two callback calls.
- Acceptance: matches M independent `TurboCKF` filters to 1e-10; for M=10,000 with a
  4-state model, at least 20x faster than looping M filters (report numbers).

## B11. Filter consistency checks (NEES/NIS)

- Need: tuning Q and R means checking filter consistency. `FilterRun` already has NIS.
- API: `turbo_ckf.consistency(run, truth=None, alpha=0.05)` returns the average NIS with
  its two-sided chi-square acceptance interval (`dim_z` dof, N samples), and NEES with
  its interval when `truth (N, dim_x)` is given.
- Acceptance: on a correctly tuned linear-Gaussian simulation both statistics fall
  inside their 95% intervals in at least 90 of 100 seeds; with R mis-scaled by 4x, NIS
  falls outside in at least 90 of 100. The chi-square quantiles need no SciPy: use a
  Wilson-Hilferty approximation, or ask the person before adding a dependency.

## B12. `reset_jitter_counters()` on `TurboCKF`

- Need: after PR #25, `TurboSRCKF` counts seed-time jitter, and both filters keep
  counters across `copy()`/`from_dict()`. Users who want per-step counts need to zero
  them after seeding, but only `TurboSRCKF` has a Python-level
  `reset_jitter_counters()`. The `TurboCKF` backend already has the method.
- API: `TurboCKF.reset_jitter_counters()`, with the same behavior as the SR one (zero
  `last_jitter`, `max_jitter`, `jitter_count`, `singular_innovation_count`), going
  through `_sync()`/`_stepped(())` like any other backend call.
- Acceptance: the counters read 0 afterwards; the filter state is unchanged
  (bit-identical x and P); the next step counts normally; a test for each.

# turbo-ckf review

## Review log

| Date | Commit | Version | Scope |
| ---- | ------ | ------- | ----- |
| 2026-09-24 | 807ec67 | 0.8.0 | First full review. All of `src/lib.rs`, `turbo_ckf/*.py`, README, CHANGELOG, CI/release workflows. Findings verified by running code on macOS arm64, Python 3.12.0, NumPy 2.4.2, FilterPy 1.4.5. |
| 2026-09-24 | 807ec67..#25 | 0.9.0 | Fix pass, same day. Parallel agents fixed A1-A8 and most lower-priority items and built B1, in PRs #20-#25. Each fix was re-verified against the original repro before merge. A8 was found during the pass. |

No earlier review file existed. An internal `AUDIT.md` was referenced before commit
0331709 but was never committed, so there was nothing to build on.

Baseline at this commit: `pytest turbo_ckf_tests` gives 190 passed, 100% Python line
coverage. None of the bugs below is caught by the suite: the relevant tests either use
constant matrices, only the easy direction of a scenario, or assert on a counter that
does not see the failure.

## Package summary (yardstick for priorities)

Rust-backed Cubature Kalman Filter with a FilterPy-shaped Python API, plus the KCKF AHRS
equations from arXiv:2602.12283. The value proposition is inferred (no single statement
in the repo): faster than FilterPy for predict/update loops, numerically honest
diagnostics, and linear batch, parallel and smoothing paths. Users are Python sensor
fusion, tracking and orientation-estimation engineers and researchers, many migrating
from FilterPy, plus AHRS users following the paper. Silent wrong results rank above
loud failures; the headline paths (per-step CKF, AHRS, `batch_filter`/`rts_smooth`,
`TurboSRCKF`) rank above opt-in extras.

Status values: `open`, `in progress`, `fixed`. "Verified" means reproduced by running
code in this review; "inferred" means read from code or docs but not executed.

## Part A: problems to fix

### A1. `rts_smooth` uses the wrong transition for time-varying models (off by one vs `batch_filter` and FilterPy)

- Status: fixed in PR #20 (2026-09-24). Length-N `Fs`/`Qs` now use `Fs[k+1]`; regression test compares against FilterPy under irregular dt.
- Severity: high
- Where: `src/lib.rs` `rts_smooth` loop (~L784-808, reads `fs_arr[[k,..]]`, `qs_arr[[k,..]]`
  for the k -> k+1 step); docstrings in `turbo_ckf/core.py` `TurboCKF.rts_smooth`
  (~L1183-1185, "FilterPy-compatible, last entry unused") and `TurboCKF.batch_filter`
  (~L980-983, "Per-step inputs match the contract that rts_smooth consumes").
- Evidence (verified): `batch_filter` treats `Fs[k]` as the (k-1) -> k transition.
  FilterPy's `rts_smoother` uses `Fs[k+1]`/`Qs[k+1]` for the k -> k+1 step (first entry
  unused). turbo's `rts_smooth` uses `Fs[k]` (last entry unused). With irregular `dt`,
  `batch_filter` matches FilterPy to 1.5e-14 and FilterPy's smoother matches a
  hand-written RTS to 1.6e-14, but `rts_smooth(xs, Ps, Fs, Qs)` is off by up to 3.07
  state units. `rts_smooth(xs, Ps, Fs[1:], Qs[1:])` is correct to 1.8e-15.
  `test_rts_smoother.py::test_non_constant_f_is_handled` only checks finiteness.

  ```python
  import numpy as np, turbo_ckf as tc
  rng = np.random.default_rng(0); N = 60
  dts = rng.uniform(0.05, 1.0, N)
  Fs = np.array([[[1, d], [0, 1]] for d in dts])
  Qs = np.array([0.3 * np.array([[d**3 / 3, d**2 / 2], [d**2 / 2, d]]) for d in dts])
  zs = rng.normal(size=(N, 1)).cumsum(axis=0)
  xs, Ps, _ = tc.batch_filter(np.zeros(2), 10 * np.eye(2), zs, Fs, [[1.0, 0.0]], Qs, 0.5 * np.eye(1))
  a, _ = tc.rts_smooth(xs, Ps, Fs, Qs)          # documented composition
  b, _ = tc.rts_smooth(xs, Ps, Fs[1:], Qs[1:])  # FilterPy-equivalent
  print(np.abs(a - b).max())                    # 0.928, should be ~0
  ```
- Why it matters: smoothed trajectories look plausible but are wrong whenever `F` or `Q`
  varies per step (irregular sampling, time-varying models). Constant-matrix users are
  unaffected, which is why tests pass. The docs explicitly promise FilterPy
  compatibility and one-call composition with `batch_filter`.
- Fix: for length-N input use `Fs[k+1]`/`Qs[k+1]` (matching FilterPy and `batch_filter`);
  keep length N-1 input as "entry k is the k -> k+1 transition". Correct both docstrings
  and add a regression test comparing against `filterpy.kalman.KalmanFilter.rts_smoother`
  with per-step `Fs`/`Qs`. This changes results for anyone who adapted to the current
  convention, so call it out in the CHANGELOG.

### A2. `update_paper_ahrs` silently needs unit-normalized accelerometer and magnetometer input

- Status: fixed in PR #20 (2026-09-24). Rust `update_paper_ahrs` normalizes both 3-vectors; the recorded `z` is the normalized one. The callback path (`update(z, hx=observation_model, ...)`) still takes z as given; see the lower-priority list.
- Severity: high
- Where: `src/lib.rs` `update_paper_ahrs` (~L261-328) uses `z` directly in
  `y = z - z_pred`; `paper_observation_model` predicts unit vectors;
  `magnetic_reference_terms` (~L550) normalizes internally, only to compute m_N/m_D.
  `turbo_ckf/core.py` `update_paper_ahrs` (~L645-662). README AHRS section says nothing
  about units.
- Evidence (verified): static attitude, 300 steps. Unit-normalized `z` converges to a
  0.15 deg error; the same readings in m/s^2 and uT (scaled by 9.81 and 48) end at
  124 deg error with NIS ~1e5, and no error or warning is raised.

  ```python
  import numpy as np
  from turbo_ckf import TurboCKF, normalize_quaternion, observation_model
  q_true = normalize_quaternion([0.9, 0.2, -0.3, 0.25])
  z_unit = observation_model(q_true, 0.45, 0.89)
  for scale in ((1.0, 1.0), (9.81, 48.0)):
      kf = TurboCKF(4, 6, 0.01, hx=observation_model, fx=lambda s, dt: s)
      kf.x = [1, 0, 0, 0]; kf.P = 0.1; kf.Q = 1e-6
      z = np.r_[scale[0] * z_unit[:3], scale[1] * z_unit[3:]]
      for _ in range(300):
          kf.predict_linear_model(np.eye(4)); kf.update_paper_ahrs(z, 1e-2, 1e-2)
          kf.normalize_state_quaternion_backend()
      print(scale, 2 * np.degrees(np.arccos(min(1.0, abs(kf.x @ q_true)))))
  ```
- Why it matters: the AHRS path is the package's research basis and a README headline.
  Raw IMU readings are the natural input, and they produce garbage attitudes silently.
  `sigma_acc2`/`sigma_mag2` are also only meaningful in unit-vector space, which is not
  documented either.
- Fix: normalize `z[:3]` and `z[3:]` inside the Rust `update_paper_ahrs` (it already
  computes both norms), and document that `sigma_*2` are variances of the normalized
  vectors. Add a test that raw-unit and unit input give the same posterior.

### A3. `TurboSRCKF` re-factors P on every call and hides the jitter it adds

- Status: fixed in PR #25 (2026-09-24), as part of B1. `chol_P` stays in Rust across steps; P/Q/R are factored only on assignment or after an in-place edit, and every in-loop jitter counts. The ill-conditioned test now asserts `max_jitter == 0`.
- Severity: high
- Where: `turbo_ckf/core.py` `TurboSRCKF.predict`/`update` call `_push_state_to_backend`
  (~L1679-1680) before every Rust call; `src/lib.rs` `SquareRootCubatureKalmanFilter::set_state`
  (~L1343-1373) runs `stable_cholesky` on P, Q and R and deliberately does not bump
  `jitter_count`. The snapshot hands back `P = chol_P chol_P^T`, which is pushed and
  re-factored on the next call.
- Evidence (verified): in the exact setup of
  `test_sr_ckf.py::test_srckf_jitter_count_stays_zero_on_ill_conditioned_run`, jitter was
  added on 5000 of 5000 steps (`max_jitter` 1e-10) while `jitter_count` stayed 0, so the
  test passes without testing its claim. On a well-conditioned 4-state model,
  `TurboSRCKF` costs 18.3 us per predict+update vs 15.4 us for `TurboCKF`.

  ```python
  import numpy as np
  from turbo_ckf import TurboSRCKF
  def fx(s, dt):
      f = np.eye(4); f[0, 2] = dt; f[1, 3] = dt
      return s @ f.T
  rng = np.random.default_rng(5)
  U = np.linalg.qr(rng.standard_normal((4, 4)))[0]
  kf = TurboSRCKF(4, 2, 0.001, hx=lambda s: s[:, :2], fx=fx)
  kf.P = U @ np.diag([1e6, 1e6, 1e-12, 1e-12]) @ U.T; kf.Q = 1e-14; kf.R = 1e-10
  kf.reset_jitter_counters(); hits = 0
  for _ in range(5000):
      kf.predict(); kf.update(np.zeros(2)); hits += kf.last_jitter > 0
  print(kf.jitter_count, kf.max_jitter, hits)   # 0 1e-10 5000
  ```
- Why it matters: the class exists to never re-decompose P ("silent jitter is
  structurally impossible"). In practice it squares the condition number every step (P
  formed in Python, re-Choleskyed in Rust), so it loses the SR numerical advantage, runs
  slower than the plain CKF, and reports zero jitter while adding it.
- Fix: keep `chol_P` in Rust as the source of truth. Push x/P/Q/R only when the user
  assigned them (dirty flags in the property setters), or add a `set_state_factor` that
  takes `chol_P` directly. Count every in-loop `stable_cholesky` jitter in `jitter_count`.
  Rewrite the test to assert `max_jitter == 0` after `reset_jitter_counters()`.
  Item B1 fixes the root cause for both classes.

### A4. Adaptive R estimation produces an indefinite R and crashes the filter when R or P starts overestimated

- Status: fixed in PR #22 (2026-09-24). The R channel uses the residual form `R + R S^-1 (y y^T - S) S^-1 R` (Akhlaghi et al. 2017) with eigenvalue projection on write-back, and uses the R actually applied. Projection alone stopped the crash but let up to 12/20 runs diverge. After the fix: 0 crashes, 0 divergences in all scenarios.
- Severity: high
- Where: `turbo_ckf/core.py` `_AdaptiveNoiseEstimator.step` (~L311-342) and `_clamp_pd`
  (~L353-358), which only floors the diagonal; `enable_adaptive_noise` docstring
  (~L753-755) claims the floor keeps the covariances positive-definite.
- Evidence (verified): 4-state CV model, `dim_z=2`, default settings
  (`window=30, mode="R", alpha=0.3`), 20 seeds each:
  - P0=100, R0=1, true R=0.01: indefinite R and `RuntimeError: unable to compute stable
    Cholesky factor` in 20/20 runs;
  - P0=10, R0=1, true R=0.25: 14/20 crash;
  - P0=1, R0=4, true R=0.25, alpha=0.05: 20/20 crash;
  - P0=1, R0=1, true R=1 (well tuned): 0/20.
  The same runs without adaptive noise all complete. Example estimate after flooring:
  `[[1e-12, 0.0155], [0.0155, 1e-12]]` (eigenvalues -0.0155, +0.0155).
  `test_adaptive_noise.py` only covers `dim_z=1` with R underestimated (R0 = true/4),
  where the off-diagonal failure cannot occur.

  ```python
  import numpy as np
  from turbo_ckf import TurboCKF
  def fx(s, dt):
      f = np.eye(4); f[0, 2] = dt; f[1, 3] = dt
      return s @ f.T
  kf = TurboCKF(4, 2, 0.1, hx=lambda s: s[:, :2], fx=fx)
  kf.P = 1.0; kf.Q = 1e-4; kf.R = 4.0      # true R is 0.25
  kf.enable_adaptive_noise()
  rng = np.random.default_rng(0)
  for _ in range(500):
      kf.predict(); kf.update(rng.normal(0, 0.5, 2))   # raises RuntimeError
  ```
- Why it matters: starting with a conservative (overestimated) R or large P is standard
  practice, and it is exactly when a user reaches for adaptive R. The feature crashes in
  that case. Early on, `y y^T - H P H^T` has negative diagonals, which get floored to 1e-12
  while the off-diagonals survive.
- Fix: project the estimate onto the PSD cone (symmetric eigendecomposition, clip
  eigenvalues at `diagonal_floor`), or switch to the residual-based form
  `R = (1-a) R + a (e e^T + H P_post H^T)` with `e = z - h(x_post)`, which is PSD by
  construction. Add `dim_z >= 2` tests with R and P overestimated. Secondary issue in the
  same code (inferred, not run): `_apply_adaptive_noise` passes `R_current=self.R` while
  `S` was built from a per-call `R` override (`update(z, R=...)`, `run(Rs=...)`), so the
  estimate is biased by `self.R - R_used`. Either use the R actually applied or skip
  adaptation when an override is passed.

### A5. NaN/inf gets into the state silently on the batch paths and through callback outputs

- Status: fixed in PR #23 (2026-09-24). `batch_filter` validates all inputs up front; `batch_parallel_step` has status 3 for non-finite per-filter input; callback output is checked in Rust `call_model_vectorized` before x/P are touched.
- Severity: medium
- Where: `src/lib.rs` `batch_filter_linear` (~L847-976) and `batch_parallel_step`
  (~L1105-1237) have no finiteness check on `zs` (or on F/H/Q/R); `call_model_vectorized`
  (~L448-472) and `turbo_ckf/core.py` `_apply_model` (~L1313-1339) check callback output
  shape but not finiteness.
- Evidence (verified): `batch_filter` with one NaN in 10 observations returns 7 of 10
  rows as NaN and raises nothing. `batch_parallel_step` with a NaN observation returns
  `status = 0` ("ok") for that filter alongside a NaN state. An `fx` that returns NaN
  leaves `x = [nan, nan]` after `predict()`; the next `update()` fails with the
  unhelpful `RuntimeError: unable to compute stable Cholesky factor`.
- Why it matters: v0.8.0 added a NaN guard to `update()` precisely because this kind of
  corruption is hard to trace, but `batch_filter` (the recommended linear path) and
  `batch_parallel_step` (whose status code promises per-filter health) skipped it.
- Fix: validate `zs` (and matrices) in Rust for both batch functions: raise for
  `batch_filter` (or treat all-NaN rows as missing, see B4); set status `3`
  ("non-finite input, update skipped") in `batch_parallel_step`. Check callback outputs
  with `np.isfinite` in `_apply_model` and raise a message naming `fx` or `hx`.

### A6. Standard motion models assume an undocumented blocked state layout

- Status: fixed in PR #21 (2026-09-24). `layout="blocked" | "interleaved"` on both standard-model predicts, documented in README and docstrings.
- Severity: medium
- Where: `src/lib.rs` `transition_matrix` (~L474-508); `turbo_ckf/core.py`
  `predict_standard_model[_ckf]` (~L565-581). The layout `[pos..., vel...]` /
  `[pos..., vel..., acc...]` only appears in a Rust error message for odd `dim_x`.
- Evidence (verified): FilterPy-style interleaved state `[x, vx, y, vy] = [0, 1, 100, 0]`
  after one `predict_standard_model("constant_velocity")` with dt=1 gives
  `x = [100, 1, 100, 0]` (x picked up y); the correct interleaved result is `[1, 1, 100, 0]`.
  No error is raised. `dim_x=2` is unaffected, which is why the quickstart works.
- Why it matters: FilterPy migrants, the main audience, commonly use the interleaved
  layout for 2-D/3-D tracking and get silently wrong predictions on the fastest path.
- Fix: document the layout in README and both docstrings, and add a
  `layout="blocked" | "interleaved"` argument (or a separate model name) so both
  conventions work. See B6.

### A7. `copy()` / `from_dict()` lose diagnostic counters; no pickle support; `TurboSRCKF` cannot be copied at all

- Status: fixed in PR #25 (2026-09-24), as part of B1. `copy()` clones the Rust object, `to_dict()`/`from_dict()` carry every backend field including counters, both classes pickle via `__reduce__`, and `TurboSRCKF` has `copy`/`to_dict`/`from_dict`/`__deepcopy__`.
- Severity: medium
- Where: `turbo_ckf/core.py` `TurboCKF.copy` (~L859-889) and `from_dict` (~L924-962)
  build a fresh Rust backend (counters 0) and only set the Python-side counters, which
  `_pull_state_from_backend` (~L1249-1278) overwrites on the next step. Neither class
  defines `__getstate__`/`__reduce__`. `TurboSRCKF` has no `copy`, `to_dict`,
  `from_dict` or `__deepcopy__`.
- Evidence (verified): original filter `jitter_count` 1; its copy reads 1, then 0 after
  one step; a `from_dict` restore also reads 0 after one step.
  `pickle.dumps(TurboCKF(...))` and `pickle.dumps(TurboSRCKF(...))` raise
  `TypeError: cannot pickle 'builtins.CubatureKalmanFilter' object` (and the SR
  equivalent); `copy.deepcopy(TurboSRCKF(...))` raises the same.
- Why it matters: `copy()` is documented as "useful for Monte-Carlo runs", but copied
  filters under-report jitter and singular innovations, and filters cannot be sent to
  `multiprocessing`/`joblib` workers or checkpointed with pickle.
- Fix: add a backend method to set counters (or include them in `set_state`) and call it
  from `copy`/`from_dict`/`reset`. Implement `__getstate__`/`__setstate__` on top of
  `to_dict()` (callbacks are pickled by reference, so module-level `fx`/`hx` work). Give
  `TurboSRCKF` the same `copy`/`to_dict`/`from_dict`/`__deepcopy__` surface.

### A8. `TurboCKF` cubature moments cancel catastrophically when the state is far from zero

- Status: fixed in PR #24 (2026-09-24). Moments are built from deviations about the mean; the ECEF repro runs 200 steps and matches the same filter at the origin to about 2 ulp. The NaN-NIS and jitter-on-raising-callback fixes went in the same PR.
- Severity: high
- Where: `src/lib.rs` `CubatureKalmanFilter::predict_custom`, `predict_ckf_linear`,
  `update`, `update_paper_ahrs` compute covariances as `E[x x^T] - mean mean^T` and
  `Pxz = sigma^T Z / 2n - x z_pred^T` (uncentred).
- Evidence (verified): 1-D constant-velocity filter with `x0 = [6.4e6, 0]` (Earth radius
  in metres), `P0 = diag(100, 1)`, `Q = diag(1e-4, 1e-4)`, `R = 1e-2`, `dt = 1`.
  `TurboCKF` raises "unable to compute stable Cholesky factor" at step 8. The same run
  at offset 0 is fine, and `TurboSRCKF` (which centres its sigma points) runs it for
  200 steps.
- Why it matters: GNSS/ECEF positions, geodetic coordinates in metres and epoch
  timestamps all put |x| far above sqrt(P). Those users get a crash, or a quietly
  wrong P before the crash.
- Fix: centre the sigma points before forming the moments (`dev = X - mean`,
  `cov = dev^T dev / 2n + Q`, and likewise for Pzz/Pxz). Two related changes go in the
  same functions: `update_likelihood_terms` uses `mahal2.max(0.0)`, which turns NaN
  into 0; and `record_jitter` runs before the callback, so a raising callback still
  changes the jitter counters.

### Lower-priority issues noted (not in the top list)

- CI runs the Python tests on Ubuntu only, but wheels ship for macOS and Windows too.
  There is no `cargo test` job and no Rust unit tests, although the `pyproject.toml`
  coverage comment says "the Rust backend is covered by the cargo test/clippy jobs".
  Status: partly fixed in PR #21 (macOS and Windows test job on Python 3.12; the
  coverage comment is accurate now). Rust unit tests and `cargo test` remain open as
  backlog item B9.
- `stable_cholesky` uses absolute jitter (1e-12 up to 1e-6) regardless of P's scale: too
  small to rescue large-magnitude P, and a large relative distortion for tiny P (for
  example quaternion covariances). Status: fixed in PR #23. Verified during the fix:
  failures start near 1e10 (1e-6 is under half an ulp above ~1.7e10), and a two-clock
  example in seconds gave NIS 0.083 instead of 0.25. Jitter is now relative to each
  diagonal entry.
- Stale commands in docs: `CONTRIBUTING.md` still uses `maturin develop --release -m
  pyproject.toml`, which CHANGELOG 0.8.0 says fails on current maturin;
  `turbo_ckf/setup_env.sh` prints a `unittest discover` command that silently skips the
  pytest-style tests (`test_sr_ckf.py`, `test_benchmark_pytest.py`). Status: fixed in
  PR #21.
- Paper attribution differs: README cites "Shing, Y. C., et al.", CHANGELOG 0.1.0 cites
  "Yamagishi and Jing" for the same arXiv ID. Status: fixed in PR #21. The arXiv page
  lists Shunsei Yamagishi and Lei Jing, "A Lightweight Cubature Kalman Filter for
  Attitude and Heading Reference Systems Using Simplified Prediction Equations", IEEE
  Access 14 (2026); the README was wrong and now matches.
- Dependencies are 2023-era (PyO3 0.20, rust-numpy 0.20, nalgebra 0.32). Tests pass on
  NumPy 2.4.2 here, but official NumPy 2 support in rust-numpy arrived in 0.21 (inferred
  from the rust-numpy changelog, not re-checked in this review). The crate also needs
  `#![allow(non_local_definitions)]` for PyO3 0.20 macros. Status: open, backlog item B8.
- `TurboSRCKF.update` raises `RuntimeError` on a singular innovation factor, while
  `TurboCKF` falls back to a pseudo-inverse and counts it. Inconsistent, not necessarily
  wrong. Status: open (no change made).
- Found during the fix pass: the AHRS callback path
  (`update(z, hx=observation_model, hx_args=(m_n, m_d))`) and the NumPy helpers in
  `paper_ahrs.py` still take z as given, so raw units are still wrong there. Only the
  Rust `update_paper_ahrs` normalizes. Status: open; a docstring note on
  `observation_model` would cover it.
- Found during the fix pass: the adaptive-noise default `alpha=0.3` averages about six
  updates, so single-step R estimates are noisy (median error ~72%, worst 3.3x, in the
  PR #22 scenarios). Time-averaged estimates are fine. Status: open; consider a smaller
  default such as 0.05.
- Found during the fix pass (not investigated): in `test_math.py`, `TurboCKF` P differs
  from FilterPy's by a steady 1.000e-3, exactly the Q used, both before and after the
  PR #24 centring change. That is why that test uses a 1.1e-3 P tolerance. It may be a
  convention difference in where Q enters (prior vs posterior comparison) rather than a
  bug. Status: open; worth a look before claiming FilterPy parity for P.
- Behavior note from PR #25: `TurboSRCKF` now counts seed-time jitter (assigning a
  degenerate P, or Q = 0) in `jitter_count`, so a fresh filter can report
  `jitter_count == 1` before any step. Call `reset_jitter_counters()` after seeding if
  you only want per-step jitter. There is no Python-level
  `TurboCKF.reset_jitter_counters()` yet (the backend has one). Status: open.

## Part B: features to add or extend

Agent-ready specs for these (and further items B7-B11) are in `docs/FEATURES.md`. That
file is the one to hand an agent; keep statuses in both files in sync.

### B1. Keep filter state in Rust and expose lazy views; run the `run()` loop in Rust

- Status: done in PR #25 (2026-09-24). Measured: 12.6 -> 4.1 us/step (standard model + update), callback path 15.7 -> 6.8, `TurboSRCKF` 18.9 -> 6.6, `run()` 10k steps 182 -> 51 ms. `run()` stayed a Python loop: the raw backend calls cost 4.1 us/step vs 5.0 for `run()`, so a Rust loop would save at most ~18%.
- Effort: medium
- Serves: the core promise of high-throughput predict/update, and fixes the root cause of A3.
- Evidence (verified): for a 4-state/2-measurement model, one wrapper
  `predict_standard_model` + `update` step costs 12.3 us. The Rust math is ~0.3 us
  (predict) + ~1.6 us (update, including the Python `hx` callback). Push (`set_state`,
  0.4 us) plus pull (`snapshot` + 22 `np.asarray`, 4.0 us), done twice per step, is about
  71% of the step.
- Why high priority: it is the largest available speedup on the API every user calls
  (roughly 3-5x on small models), and it makes `TurboSRCKF` keep its factor.
- Sketch: make `x`/`P`/`Q`/`R`/diagnostics properties that read from the backend on
  access (cache per step) and push only on assignment. In-place edits such as
  `kf.x[0] = 1` then need a documented rule (return copies, or write-through views).
  Move `_run_filter` into Rust so a sequence costs one crossing per callback instead of
  four array round trips per step. Keep FilterPy-style attribute names.

### B2. Wheels for aarch64 Linux and Intel macOS; test on every release platform

- Status: open
- Effort: small
- Serves: embedded and robotics users (the paper benchmarks on a Raspberry Pi 4) and the
  README claim "Wheels ... on Linux, macOS, and Windows".
- Evidence (verified via the PyPI JSON API): v0.8.0 ships `macosx_11_0_arm64`,
  `manylinux_2_17_x86_64` and `win_amd64` wheels plus an sdist. On a Raspberry Pi,
  Jetson or Graviton host, `pip install turbo-ckf` falls back to the sdist and needs a
  Rust toolchain.
- Sketch: add `target: aarch64` (manylinux, via QEMU or an arm64 runner) and
  `macos-13`/`x86_64-apple-darwin` entries to `release.yml`; add macOS, Windows and arm64
  jobs to the `ci.yml` test matrix. Since the wheels are abi3, one wheel per platform is
  enough.

### B3. One-call AHRS pipeline in Rust

- Status: open
- Effort: medium; depends on A2
- Serves: the paper-derived AHRS use case, whose selling point is speed.
- Why high priority: today each AHRS step is four wrapper calls (`Q` from
  `process_noise_from_quaternion` in NumPy, `predict_linear_model`, `update_paper_ahrs`,
  normalize), each paying the B1 sync cost. `benchmark_paper.py` already shows the
  wrapper erasing most of the KCKF-vs-CKF gain (1.67x in the backend vs 1.03x through the
  wrapper, per README).
- Sketch: `ahrs_filter(q0, P0, gyro (N,3), acc (N,3), mag (N,3), dts, gyro_var,
  sigma_acc2, sigma_mag2)` returning quaternions, covariances and NIS, all in Rust:
  F from gyro, Q from the quaternion, normalized update, renormalization. Add an
  accel-only update for when the magnetometer is disturbed or missing, and optional NIS
  gating of the mag channel.

### B4. Missing-data support and diagnostics in `batch_filter`

- Status: open
- Effort: small; pairs with A1 and A5
- Serves: offline processing of real sensor logs on the fastest linear path.
- Why high priority: `run()` handles dropouts (`None` or all-NaN rows), but the ~22x
  faster `batch_filter` corrupts its output on any NaN (A5). Users with real logs are
  pushed back to the slow path.
- Sketch: treat all-NaN rows as "predict only" in `batch_filter_linear`; return priors,
  NIS and a per-step status or singular-innovation flag alongside `(xs, Ps, lls)`,
  ideally as a `FilterRun` so its output matches `run()`.

### B5. Nonlinear (cubature) RTS smoother

- Status: open
- Effort: medium
- Serves: offline trajectory reconstruction for the package's namesake nonlinear CKF.
- Why high priority: `rts_smooth` is linear-only (needs `Fs`), so users of `predict()`
  with a nonlinear `fx` cannot smooth at all. Smoothing is a routine offline need and
  FilterPy users expect it (FilterPy has UKF RTS).
- Sketch: during `run()`, store the predict cross-covariance `P_{k,k+1}` from the
  propagated cubature points (Rust already has them in `predict_custom`), then do a
  cubature RTS backward pass in Rust. Expose it as `kf.run(..., smooth=True)` or
  `TurboCKF.ckf_rts_smooth(run_result)`.

### B6. FilterPy migration helpers

- Status: open
- Effort: small; overlaps A6
- Serves: FilterPy migrants, the main audience (API names, parity tests and benchmarks
  are all framed against FilterPy).
- Why high priority: the fast standard-model path silently assumes a different layout
  (A6), provides no matching `Q`, and cannot take a per-call `dt`. These are cheap to fix
  and remove the most likely wrong-result traps during migration.
- Sketch: a `Q_discrete_white_noise`-style helper for the built-in CV/CA models (both
  layouts), a `layout` argument, a `dt=` argument on `predict_standard_model[_ckf]`, and
  a README section mapping common FilterPy calls (`KalmanFilter`, `UnscentedKalmanFilter`,
  `batch_filter`, `rts_smoother`) to turbo-ckf equivalents, including the vectorized
  callback contract and `Fs` indexing.

### Also considered, ranked lower

- Nonlinear parallel bank (`batch_parallel_step` for callback models, one vectorized
  callback over M x 2n sigma points). Large effort, narrower audience.
- Joseph-form covariance update option for the CKF `update`, for long runs with tiny R.
- `TurboSRCKF` parity with `TurboCKF`: linear/standard-model predicts, AHRS update,
  adaptive noise.
- NEES/NIS consistency-test helpers on top of `FilterRun`.

## Areas not covered or covered lightly

- The paper itself was not read. The AHRS equations were checked for internal
  consistency (the observation model is a proper rotation) and for convergence in a
  static scenario only; sign and frame conventions against the paper are unverified.
- Test files other than `test_sr_ckf.py`, `test_rts_smoother.py`, `test_adaptive_noise.py`
  (scenario setup only) and the two AHRS test files were not read; they were only run.
- Benchmark scripts (`benchmark.py`, `benchmark_paper.py`, `verify_before_after.py`)
  were not reviewed, and README benchmark numbers were not re-measured. The
  pytest-benchmark run showed about 23 ms for turbo's standard model vs 92 ms for
  FilterPy on its workload.
- Only macOS arm64 was exercised. Linux and Windows wheels, the release workflow, and
  sdist builds were not run.
- The SR-CKF update math (QR innovation factor, rank-1 downdates) was checked by
  reading, and the downdate rotation algebra is correct, but `dim_z > dim_x` and the
  downdate-fallback branch were not exercised.
- `stable_cholesky` behavior at extreme covariance scales, thread safety of sharing one
  filter across threads, and memory use of `batch_filter`/`batch_parallel_step` at large
  N or M were not tested.

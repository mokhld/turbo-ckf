"""Parallel batch step correctness + speedup tests.

The parallel batch step takes a bank of M independent linear KFs sharing
(F, H, Q, R), with per-filter (x, P, z), and advances every filter by one
predict + update in parallel. Distinct from batch_filter (one filter,
many observations).

Two marquee checks:
  1. Numerical parity at ~1e-10 on M=100 vs a sequential Python loop
     calling TurboCKF.predict_linear_model + update.
  2. At M=10,000 the parallel step is >= 4x faster than the same Rust
     per-step path called sequentially.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from turbo_ckf import TurboCKF, batch_parallel_step


def _cv_matrices(dt: float, q_var: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.array([[1.0, dt, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, dt],
                  [0.0, 0.0, 0.0, 1.0]], dtype=float)
    # Q from white-noise acceleration: G * G^T * q_var, with G the
    # one-step velocity-impulse map.
    g = np.array([[0.5 * dt * dt, 0.0],
                  [dt,           0.0],
                  [0.0,          0.5 * dt * dt],
                  [0.0,          dt]], dtype=float)
    q = q_var * (g @ g.T)
    return f, q


def _hx_2d_position() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]], dtype=float)


def _make_bank(M: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random (x, P, z) bank for M filters. P is PD by construction
    (P = A * A^T + jitter * I), so every filter is well-conditioned."""
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((M, 4))
    A = rng.standard_normal((M, 4, 4)) * 0.3
    Ps = np.einsum("mij,mkj->mik", A, A) + 0.5 * np.eye(4)
    zs = rng.standard_normal((M, 2))
    return xs, Ps, zs


def _sequential_reference(
    xs: np.ndarray,
    Ps: np.ndarray,
    zs: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loop a TurboCKF per-step over the M filters. The parallel batch
    step must agree with this to ~1e-10."""
    M, dim_x = xs.shape
    dim_z = zs.shape[1]

    def hx_linear(sigma: np.ndarray) -> np.ndarray:
        return (H @ sigma.T).T

    def fx_cv(sigma: np.ndarray, dt: float) -> np.ndarray:
        # fx is unused on the linear path; predict_linear_model uses F
        # directly. Provide a no-op stub that satisfies the constructor.
        return sigma

    xs_out = np.zeros_like(xs)
    Ps_out = np.zeros_like(Ps)
    lls_out = np.zeros(M)
    for i in range(M):
        kf = TurboCKF(dim_x=dim_x, dim_z=dim_z, dt=1.0, hx=hx_linear, fx=fx_cv)
        kf.x = xs[i].copy()
        kf.P = Ps[i].copy()
        kf.Q = Q.copy()
        kf.R = R.copy()
        kf.predict_linear_model(F)
        kf.update(zs[i])
        xs_out[i] = kf.x
        Ps_out[i] = kf.P
        lls_out[i] = kf.log_likelihood
    return xs_out, Ps_out, lls_out


class ParallelBatchStepParityTests(unittest.TestCase):
    """The marquee test: parallel batch step matches a per-step
    sequential loop to ~1e-10 on every filter in the bank."""

    def test_matches_sequential_loop_M100(self):
        M = 100
        F, Q = _cv_matrices(dt=0.1, q_var=0.5)
        H = _hx_2d_position()
        R = 0.25 * np.eye(2)
        xs, Ps, zs = _make_bank(M, seed=0)

        ref_xs, ref_Ps, ref_lls = _sequential_reference(xs, Ps, zs, F, H, Q, R)
        new_xs, new_Ps, new_lls, status = batch_parallel_step(
            xs, Ps, zs, F=F, H=H, Q=Q, R=R
        )

        self.assertEqual(status.shape, (M,))
        self.assertEqual(int(status.max()), 0,
                         msg="well-conditioned bank should have status all 0")
        self.assertTrue(np.allclose(new_xs, ref_xs, atol=1e-10),
                        msg=f"max state diff = {float(np.max(np.abs(new_xs - ref_xs))):.2e}")
        self.assertTrue(np.allclose(new_Ps, ref_Ps, atol=1e-10),
                        msg=f"max P diff = {float(np.max(np.abs(new_Ps - ref_Ps))):.2e}")
        # Log-likelihoods agree to ~1e-9 (one more multiply away from the
        # state diff via mahal2).
        self.assertTrue(np.allclose(new_lls, ref_lls, atol=1e-9),
                        msg=f"max ll diff = {float(np.max(np.abs(new_lls - ref_lls))):.2e}")


class ParallelBatchStepShapeTests(unittest.TestCase):
    def test_shapes_and_dtypes(self):
        M = 8
        F, Q = _cv_matrices(0.1, 0.2)
        H = _hx_2d_position()
        R = 0.1 * np.eye(2)
        xs, Ps, zs = _make_bank(M, seed=1)
        new_xs, new_Ps, lls, status = batch_parallel_step(
            xs, Ps, zs, F=F, H=H, Q=Q, R=R
        )
        self.assertEqual(new_xs.shape, (M, 4))
        self.assertEqual(new_Ps.shape, (M, 4, 4))
        self.assertEqual(lls.shape, (M,))
        self.assertEqual(status.shape, (M,))
        self.assertEqual(new_xs.dtype, np.float64)
        self.assertEqual(status.dtype, np.int64)
        # All Ps should be symmetric.
        for i in range(M):
            self.assertLess(float(np.max(np.abs(new_Ps[i] - new_Ps[i].T))), 1e-12)

    def test_defaults_q_zero_r_identity(self):
        xs, Ps, zs = _make_bank(3, seed=2)
        F = np.eye(4)
        H = _hx_2d_position()
        # Should not raise; R defaults to I(2), Q defaults to 0.
        new_xs, new_Ps, lls, status = batch_parallel_step(xs, Ps, zs, F=F, H=H)
        self.assertEqual(new_xs.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(new_xs)))

    def test_bad_shapes_raise(self):
        F, Q = _cv_matrices(0.1, 0.1)
        H = _hx_2d_position()
        R = 0.1 * np.eye(2)
        xs, Ps, zs = _make_bank(4, seed=3)
        # Mismatched M between xs and zs.
        with self.assertRaises(ValueError):
            batch_parallel_step(xs, Ps, zs[:2], F=F, H=H, Q=Q, R=R)
        # Wrong F shape.
        with self.assertRaises(ValueError):
            batch_parallel_step(xs, Ps, zs, F=np.eye(3), H=H, Q=Q, R=R)
        # Empty bank.
        with self.assertRaises(ValueError):
            batch_parallel_step(np.zeros((0, 4)), np.zeros((0, 4, 4)),
                                np.zeros((0, 2)), F=F, H=H, Q=Q, R=R)


class ParallelBatchStepErrorHandlingTests(unittest.TestCase):
    """One bad filter must not poison the bank."""

    def test_singular_R_marks_status_but_keeps_other_filters(self):
        # F = I, Q = 0, R = 0 with P measured-rows zero -> S exactly 0.
        M = 4
        F = np.eye(4)
        H = _hx_2d_position()
        Q = np.zeros((4, 4))
        R = np.zeros((2, 2))

        xs, Ps, zs = _make_bank(M, seed=42)
        # Filter 0: zero variance in the measured coordinates so the
        # innovation covariance S = H P H^T + R is the zero matrix.
        Ps[0] = np.diag([0.0, 1.0, 0.0, 1.0])

        new_xs, new_Ps, lls, status = batch_parallel_step(
            xs, Ps, zs, F=F, H=H, Q=Q, R=R
        )
        # Filter 0's S = 0 -> Cholesky fails, try_inverse fails, pinv may
        # produce a (zero) matrix; either status >= 1 or status == 2.
        self.assertGreaterEqual(int(status[0]), 1)
        # Other filters stay healthy.
        for i in range(1, M):
            self.assertEqual(int(status[i]), 0,
                             msg=f"filter {i} unexpectedly reported status {int(status[i])}")
        self.assertEqual(new_xs.shape, (M, 4))


class ParallelBatchStepSpeedupTests(unittest.TestCase):
    """At M=10,000 the parallel step must beat the same per-step Rust
    path called sequentially by >= 4x. We compare against a sequential
    loop that goes through the *Rust* per-step API (not a Python
    reference loop), so we measure parallelism alone, not Python-vs-Rust.
    """

    def test_parallel_at_least_4x_faster_than_sequential_rust_loop(self):
        M = 10_000
        F, Q = _cv_matrices(dt=0.05, q_var=0.5)
        H = _hx_2d_position()
        R = 0.1 * np.eye(2)
        xs, Ps, zs = _make_bank(M, seed=123)

        def hx_linear(sigma: np.ndarray) -> np.ndarray:
            return (H @ sigma.T).T

        def fx_unused(sigma: np.ndarray, dt: float) -> np.ndarray:
            return sigma

        # Warm-up + time the sequential Rust per-step loop. Construct the
        # bank of M TurboCKF instances once (allocation amortized) then
        # time only the predict+update loop.
        kfs = []
        for i in range(M):
            kf = TurboCKF(dim_x=4, dim_z=2, dt=1.0, hx=hx_linear, fx=fx_unused)
            kf.x = xs[i].copy()
            kf.P = Ps[i].copy()
            kf.Q = Q.copy()
            kf.R = R.copy()
            kfs.append(kf)

        # Two timed runs each, take the min — guards against fluky first runs.
        # Parallel.
        batch_parallel_step(xs, Ps, zs, F=F, H=H, Q=Q, R=R)  # warm-up
        t_par_min = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            batch_parallel_step(xs, Ps, zs, F=F, H=H, Q=Q, R=R)
            t_par_min = min(t_par_min, time.perf_counter() - t0)

        # Sequential — drive the existing Rust per-step API. We don't
        # rebuild the TurboCKF objects between runs because we want to
        # measure the step cost, not allocation.
        # warm-up
        for kf in kfs[:200]:
            kf.predict_linear_model(F)
            kf.update(zs[0])
        t_seq_min = float("inf")
        for _ in range(2):
            t0 = time.perf_counter()
            for i, kf in enumerate(kfs):
                kf.predict_linear_model(F)
                kf.update(zs[i])
            t_seq_min = min(t_seq_min, time.perf_counter() - t0)

        speedup = t_seq_min / t_par_min
        # Report for the test log even when it passes.
        print(f"\n  [parallel speedup] M={M}  seq={t_seq_min*1e3:.1f}ms  "
              f"par={t_par_min*1e3:.1f}ms  speedup={speedup:.2f}x")
        self.assertGreaterEqual(
            speedup, 4.0,
            msg=f"parallel speedup {speedup:.2f}x < 4.0x target",
        )


if __name__ == "__main__":
    unittest.main()

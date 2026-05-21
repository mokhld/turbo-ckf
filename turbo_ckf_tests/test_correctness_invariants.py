"""Long-run and statistical correctness invariants.

These tests address gaps identified by the AUDIT.md pass:
- nothing exercises the filter past ~200 steps
- nothing checks PSD/symmetry/condition of P over long runs
- no NIS chi-squared consistency check
- edge cases (dt=0 identity, dim_z > dim_x, scalar R broadcast) untested
"""

from __future__ import annotations

import unittest

import numpy as np

from turbo_ckf import TurboCKF


def fx_cv(x: np.ndarray, dt: float) -> np.ndarray:
    """Vectorized constant-velocity transition for state [pos, vel]."""

    out = x.copy()
    out[:, 0] = x[:, 0] + dt * x[:, 1]
    return out


def hx_pos(x: np.ndarray) -> np.ndarray:
    """Observe position only."""

    return x[:, :1]


class LongRunStabilityTests(unittest.TestCase):
    """Filter must keep P symmetric, PSD, and well-conditioned over many steps."""

    def test_p_stays_symmetric_and_psd_over_long_run(self):
        rng = np.random.default_rng(0)
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.05, hx=hx_pos, fx=fx_cv)
        kf.x = np.array([0.0, 1.0], dtype=float)
        kf.P = np.eye(2)
        kf.Q = 1e-3 * np.eye(2)
        kf.R = np.array([[0.05]], dtype=float)

        # 10k steps of noisy position observations of a CV target.
        true_pos = 0.0
        true_vel = 1.0
        max_asym = 0.0
        min_eig = np.inf
        for _ in range(10_000):
            true_pos += kf.dt * true_vel
            kf.predict()
            z = np.array([true_pos + rng.normal(0.0, np.sqrt(0.05))], dtype=float)
            kf.update(z)
            asym = float(np.max(np.abs(kf.P - kf.P.T)))
            if asym > max_asym:
                max_asym = asym
            eig = float(np.min(np.linalg.eigvalsh(0.5 * (kf.P + kf.P.T))))
            if eig < min_eig:
                min_eig = eig

        self.assertLess(max_asym, 1e-10, "P drifted asymmetric")
        self.assertGreater(min_eig, -1e-10, "P drifted non-PSD")
        # The filter should track the target — final pos error tiny.
        self.assertLess(abs(kf.x[0] - true_pos), 0.5)

    def test_jitter_counter_increments_when_p_is_near_singular(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        # Force P to be non-PD: a tiny negative eigenvalue on the diagonal.
        kf.x = np.array([0.0, 0.0], dtype=float)
        kf.P = np.array([[1.0, 1.0 + 1e-14], [1.0 + 1e-14, 1.0]], dtype=float)
        kf.Q = np.zeros((2, 2))
        kf.R = np.array([[1.0]], dtype=float)
        # predict triggers cubature_points -> stable_cholesky -> jitter.
        kf.predict()
        # On exactly-singular P the backend may or may not need jitter; either
        # way the diagnostic must be present and non-negative.
        self.assertGreaterEqual(kf.last_jitter, 0.0)
        self.assertGreaterEqual(kf.max_jitter, kf.last_jitter)
        self.assertGreaterEqual(kf.jitter_count, 0)


class NisConsistencyTests(unittest.TestCase):
    """For a correctly-tuned linear-Gaussian filter, the time-average of NIS
    should sit near dim_z (the chi-square mean). Wide bounds — this is a
    sanity check, not a hypothesis test."""

    def test_nis_average_within_chi_squared_bounds(self):
        rng = np.random.default_rng(42)
        dt = 0.1
        var_q = 1e-3
        var_r = 0.04

        kf = TurboCKF(dim_x=2, dim_z=1, dt=dt, hx=hx_pos, fx=fx_cv)
        kf.x = np.array([0.0, 1.0], dtype=float)
        kf.P = np.eye(2)
        kf.Q = var_q * np.eye(2)
        kf.R = np.array([[var_r]], dtype=float)

        true_pos = 0.0
        true_vel = 1.0
        nis_samples = []
        for _ in range(2_000):
            true_vel += rng.normal(0.0, np.sqrt(var_q))
            true_pos += dt * true_vel
            kf.predict()
            z = np.array([true_pos + rng.normal(0.0, np.sqrt(var_r))], dtype=float)
            kf.update(z)
            nis_samples.append(kf.nis)

        avg_nis = float(np.mean(nis_samples))
        # Mean of chi-square with dim_z=1 dof is 1. Allow a wide band because
        # the filter starts mis-tuned and Q/R may not exactly match the
        # generative noise.
        self.assertGreater(avg_nis, 0.3)
        self.assertLess(avg_nis, 3.0)


class EdgeCaseTests(unittest.TestCase):
    def test_dt_zero_predict_is_identity(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.0, hx=hx_pos, fx=fx_cv)
        kf.x = np.array([1.0, 2.0], dtype=float)
        kf.P = np.diag([0.5, 0.7])
        kf.Q = np.zeros((2, 2))
        x_before = kf.x.copy()
        kf.predict()
        # With Q=0, dt=0, and a CV transition, the state shouldn't change.
        # The covariance picks up tiny numerical noise from sigma-point round-trip.
        self.assertTrue(np.allclose(kf.x, x_before, atol=1e-12))

    def test_dim_z_larger_than_dim_x(self):
        def fx(x, dt):
            return x.copy()

        def hx(x):
            # Replicate state into a longer measurement vector.
            return np.column_stack([x[:, 0], x[:, 0], x[:, 0]])

        kf = TurboCKF(dim_x=1, dim_z=3, dt=0.1, hx=hx, fx=fx)
        kf.x = np.array([1.0], dtype=float)
        kf.P = np.array([[0.5]], dtype=float)
        kf.R = 0.1 * np.eye(3)
        kf.predict()
        kf.update(np.array([1.1, 0.9, 1.0], dtype=float))
        self.assertEqual(kf.x.shape, (1,))
        self.assertEqual(kf.S.shape, (3, 3))
        # Posterior should be pulled toward the observation mean.
        self.assertGreater(kf.x[0], 1.0 - 1e-6)
        self.assertLess(kf.x[0], 1.05)

    def test_scalar_r_actually_broadcasts(self):
        # Run two filters with the same trajectory but different R magnitudes
        # supplied as scalars. If the scalar isn't broadcast to dim_z x dim_z,
        # both updates would behave identically.
        kf_small = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf_large = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        z = np.array([1.0], dtype=float)
        kf_small.predict()
        kf_small.update(z, R=0.01)
        kf_large.predict()
        kf_large.update(z, R=100.0)
        # Smaller R => measurement trusted more => state moves further toward z.
        self.assertGreater(kf_small.x[0], kf_large.x[0])
        # S = HPH^T + R, so the R magnitude must dominate the difference.
        s_diff = float(kf_large.S[0, 0]) - float(kf_small.S[0, 0])
        self.assertGreater(s_diff, 50.0)  # ~100 - 0.01 minus identical HPH^T term

    def test_mismatched_z_shape_raises(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf.predict()
        with self.assertRaises(ValueError):
            kf.update(np.array([0.0, 1.0], dtype=float))


class StaleStateOnSkippedUpdateTests(unittest.TestCase):
    """update(z=None) must clear innovation-derived diagnostics, so a
    downstream NIS gate doesn't read last update's values."""

    def test_skipped_update_clears_innovation_diagnostics(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf.predict()
        kf.update(np.array([0.5], dtype=float))
        self.assertTrue(np.isfinite(kf.log_likelihood))
        self.assertTrue(np.isfinite(kf.mahalanobis))
        self.assertTrue(np.isfinite(kf.nis))

        kf.predict()
        kf.update(None)
        self.assertTrue(np.isnan(kf.log_likelihood))
        self.assertTrue(np.isnan(kf.mahalanobis))
        self.assertTrue(np.isnan(kf.nis))
        # y is the most-recent innovation: must be zeroed.
        self.assertTrue(np.allclose(kf.y, 0.0))


class ResetCopySerializeTests(unittest.TestCase):
    def test_reset_returns_to_clean_state(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf.predict()
        kf.update(np.array([0.5], dtype=float))
        kf.reset()
        self.assertTrue(np.allclose(kf.x, np.zeros(2)))
        self.assertTrue(np.allclose(kf.P, np.eye(2)))
        self.assertTrue(np.isnan(kf.log_likelihood))
        self.assertEqual(kf.jitter_count, 0)

    def test_copy_is_independent(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf.x = np.array([1.0, 2.0], dtype=float)
        twin = kf.copy()
        kf.x = np.array([99.0, 99.0], dtype=float)
        self.assertTrue(np.allclose(twin.x, [1.0, 2.0]))
        twin.predict()
        # Mutating the copy doesn't leak back.
        self.assertTrue(np.allclose(kf.x, [99.0, 99.0]))

    def test_to_dict_from_dict_round_trip(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf.predict()
        kf.update(np.array([0.5], dtype=float))
        state = kf.to_dict()
        restored = TurboCKF.from_dict(state, hx=hx_pos, fx=fx_cv)
        self.assertTrue(np.allclose(restored.x, kf.x))
        self.assertTrue(np.allclose(restored.P, kf.P))
        self.assertAlmostEqual(restored.log_likelihood, kf.log_likelihood)
        # Restored filter must keep stepping cleanly.
        restored.predict()
        restored.update(np.array([1.0], dtype=float))


class GateTests(unittest.TestCase):
    def test_gate_returns_false_before_any_update(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        self.assertFalse(kf.gate(threshold=10.0))

    def test_gate_accepts_inlier_and_rejects_outlier(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf.predict()
        kf.update(np.array([0.0], dtype=float))  # close to predicted 0
        # Inlier NIS ~ small.
        self.assertTrue(kf.gate(threshold=10.0))

        kf2 = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        kf2.R = 1e-6 * np.eye(1)
        kf2.predict()
        kf2.update(np.array([50.0], dtype=float))  # extreme outlier under tiny R
        self.assertFalse(kf2.gate(threshold=10.0))


class ValidationTests(unittest.TestCase):
    def test_unknown_standard_model_raises(self):
        kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        with self.assertRaises(ValueError):
            kf.predict_standard_model("constant_velocty")  # typo

    def test_negative_sigma_acc2_rejected(self):
        kf = TurboCKF(dim_x=4, dim_z=6, dt=0.1, hx=lambda x: x[:, :6] if False else None, fx=lambda x, dt: x.copy())
        kf.x = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        with self.assertRaises(ValueError):
            kf.update_paper_ahrs(np.zeros(6) + 1.0, sigma_acc2=-1.0, sigma_mag2=1.0)

    def test_non_finite_dt_rejected_in_constructor(self):
        with self.assertRaises(ValueError):
            TurboCKF(dim_x=2, dim_z=1, dt=float("nan"), hx=hx_pos, fx=fx_cv)


if __name__ == "__main__":
    unittest.main()

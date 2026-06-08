"""Validation and error-path coverage for turbo_ckf.core.

These exercise the input-guard / defensive branches in TurboCKF, TurboSRCKF
and the _AdaptiveNoiseEstimator that the happy-path suites don't reach: bad
dimensions, non-finite parameters, shape mismatches on the batch/static
helpers, the callback-contract guards, and the small serialization /
introspection utilities (repr, deepcopy, from_dict, _coerce_args,
_stable_cholesky).
"""

import copy
import unittest

import numpy as np

from turbo_ckf import TurboCKF, TurboSRCKF


def fx_cv(x, dt):
    out = x.copy()
    out[:, 0] = x[:, 0] + dt * x[:, 1]
    return out


def hx_pos(x):
    return x[:, :1]


def _build(dim_x=2, dim_z=1, dt=0.1):
    return TurboCKF(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx_pos, fx=fx_cv)


class ConstructorValidationTests(unittest.TestCase):
    def test_non_positive_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF(dim_x=0, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        with self.assertRaises(ValueError):
            TurboCKF(dim_x=2, dim_z=0, dt=0.1, hx=hx_pos, fx=fx_cv)

    def test_non_finite_dt_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF(dim_x=2, dim_z=1, dt=float("inf"), hx=hx_pos, fx=fx_cv)

    def test_repr_reports_dims_and_backend(self):
        kf = _build()
        text = repr(kf)
        self.assertIn("TurboCKF", text)
        self.assertIn("dim_x=2", text)
        self.assertIn("dim_z=1", text)
        self.assertIn("rust", text)


class PredictUpdateValidationTests(unittest.TestCase):
    def test_predict_rejects_non_finite_dt_override(self):
        kf = _build()
        with self.assertRaises(ValueError):
            kf.predict(dt=float("inf"))

    def test_predict_propagates_bad_callback_output_shape(self):
        # fx returns the wrong number of columns -> _apply_model rejects it.
        def bad_fx(x, dt):
            return x[:, :1]

        kf = _build()
        with self.assertRaises(ValueError):
            kf.predict(fx=bad_fx)

    def test_update_propagates_bad_callback_output_shape(self):
        def bad_hx(x):
            return x  # (2*dim_x, dim_x) instead of (2*dim_x, dim_z)

        kf = _build()
        kf.predict()
        with self.assertRaises(ValueError):
            kf.update(np.array([0.0], dtype=float), hx=bad_hx)

    def test_update_rejects_wrong_R_shape(self):
        kf = _build()
        kf.predict()
        with self.assertRaises(ValueError):
            kf.update(np.array([0.0], dtype=float), R=np.eye(3))


class PaperAhrsUpdateValidationTests(unittest.TestCase):
    def test_requires_4x6_dimensions(self):
        kf = _build(dim_x=2, dim_z=1)
        with self.assertRaises(ValueError):
            kf.update_paper_ahrs(np.zeros(6), sigma_acc2=1.0, sigma_mag2=1.0)

    def test_rejects_non_positive_sigma_acc2(self):
        kf = TurboCKF(dim_x=4, dim_z=6, dt=0.1, hx=lambda x: x[:, :6], fx=lambda x, dt: x)
        kf.x = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        with self.assertRaises(ValueError):
            kf.update_paper_ahrs(np.ones(6), sigma_acc2=0.0, sigma_mag2=1.0)

    def test_rejects_non_positive_sigma_mag2(self):
        kf = TurboCKF(dim_x=4, dim_z=6, dt=0.1, hx=lambda x: x[:, :6], fx=lambda x, dt: x)
        kf.x = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        with self.assertRaises(ValueError):
            kf.update_paper_ahrs(np.ones(6), sigma_acc2=1.0, sigma_mag2=-1.0)


class QuaternionNormalizationTests(unittest.TestCase):
    def test_normalize_state_quaternion_requires_dim4(self):
        kf = _build(dim_x=2, dim_z=1)
        with self.assertRaises(ValueError):
            kf.normalize_state_quaternion()

    def test_normalize_state_quaternion_rejects_zero_norm(self):
        kf = _build(dim_x=4, dim_z=1)
        kf.x = np.zeros(4, dtype=float)
        with self.assertRaises(ValueError):
            kf.normalize_state_quaternion()

    def test_normalize_state_quaternion_unit_result(self):
        kf = _build(dim_x=4, dim_z=1)
        kf.x = np.array([0.0, 3.0, 0.0, 4.0], dtype=float)
        out = kf.normalize_state_quaternion()
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=12)
        # x_post is kept in sync with x.
        self.assertTrue(np.allclose(kf.x_post, out))

    def test_normalize_state_quaternion_backend_requires_dim4(self):
        kf = _build(dim_x=2, dim_z=1)
        with self.assertRaises(ValueError):
            kf.normalize_state_quaternion_backend()

    def test_normalize_state_quaternion_backend_unit_result(self):
        kf = _build(dim_x=4, dim_z=1)
        kf.x = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        out = kf.normalize_state_quaternion_backend()
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=12)


class SerializationAndCopyTests(unittest.TestCase):
    def test_deepcopy_is_independent(self):
        kf = _build()
        kf.x = np.array([1.0, 2.0], dtype=float)
        clone = copy.deepcopy(kf)
        clone.x[0] = 99.0
        self.assertEqual(kf.x[0], 1.0)
        self.assertEqual(clone.dim_x, kf.dim_x)

    def test_from_dict_rejects_unknown_version(self):
        kf = _build()
        state = kf.to_dict()
        state["version"] = 2
        with self.assertRaises(ValueError):
            TurboCKF.from_dict(state, hx=hx_pos, fx=fx_cv)

    def test_to_dict_from_dict_round_trip(self):
        kf = _build()
        kf.x = np.array([0.0, 1.0], dtype=float)
        kf.predict()
        kf.update(np.array([0.5], dtype=float))
        restored = TurboCKF.from_dict(kf.to_dict(), hx=hx_pos, fx=fx_cv)
        self.assertTrue(np.allclose(restored.x, kf.x))
        self.assertTrue(np.allclose(restored.P, kf.P))


class BatchFilterValidationTests(unittest.TestCase):
    def test_empty_state_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_filter(
                x0=np.zeros(0),
                P0=np.zeros((0, 0)),
                zs=np.zeros((3, 1)),
                F=np.eye(1),
                H=np.ones((1, 1)),
            )

    def test_matrix_with_bad_ndim_rejected(self):
        x0 = np.zeros(2)
        p0 = np.eye(2)
        zs = np.zeros((4, 1))
        f = np.eye(2)
        h = np.array([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            TurboCKF.batch_filter(x0, p0, zs, F=np.zeros(2), H=h)  # 1D F
        with self.assertRaises(ValueError):
            TurboCKF.batch_filter(x0, p0, zs, F=f, H=np.zeros((1, 1, 1, 1)))  # 4D H


class BatchParallelStepValidationTests(unittest.TestCase):
    def setUp(self):
        self.F = np.eye(4)
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.Q = 0.01 * np.eye(4)
        self.R = 0.1 * np.eye(2)
        self.xs = np.zeros((3, 4))
        self.Ps = np.tile(np.eye(4), (3, 1, 1))
        self.zs = np.zeros((3, 2))

    def test_xs_must_be_2d(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_parallel_step(
                np.zeros(4), self.Ps, self.zs, F=self.F, H=self.H, Q=self.Q, R=self.R
            )

    def test_zero_dim_x_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_parallel_step(
                np.zeros((3, 0)), np.zeros((3, 0, 0)), self.zs,
                F=np.zeros((0, 0)), H=np.zeros((2, 0)), Q=np.zeros((0, 0)), R=self.R,
            )

    def test_zero_dim_z_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_parallel_step(
                self.xs, self.Ps, np.zeros((3, 0)),
                F=self.F, H=np.zeros((0, 4)), Q=self.Q, R=np.zeros((0, 0)),
            )

    def test_bad_Ps_shape_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_parallel_step(
                self.xs, np.tile(np.eye(4), (2, 1, 1)), self.zs,
                F=self.F, H=self.H, Q=self.Q, R=self.R,
            )

    def test_bad_H_shape_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_parallel_step(
                self.xs, self.Ps, self.zs, F=self.F, H=np.zeros((2, 3)), Q=self.Q, R=self.R
            )

    def test_bad_Q_shape_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_parallel_step(
                self.xs, self.Ps, self.zs, F=self.F, H=self.H, Q=np.eye(3), R=self.R
            )

    def test_bad_R_shape_rejected(self):
        with self.assertRaises(ValueError):
            TurboCKF.batch_parallel_step(
                self.xs, self.Ps, self.zs, F=self.F, H=self.H, Q=self.Q, R=np.eye(3)
            )


class StaticHelperTests(unittest.TestCase):
    def test_coerce_args_variants(self):
        self.assertEqual(TurboCKF._coerce_args(None), ())
        self.assertEqual(TurboCKF._coerce_args(()), ())
        self.assertEqual(TurboCKF._coerce_args((1, 2)), (1, 2))
        self.assertEqual(TurboCKF._coerce_args([3, 4]), (3, 4))
        self.assertEqual(TurboCKF._coerce_args(7), (7,))
        arr = np.arange(3)
        coerced = TurboCKF._coerce_args(arr)
        self.assertEqual(len(coerced), 1)
        self.assertIs(coerced[0], arr)

    def test_coerce_covariance_scalar_expands_to_diagonal(self):
        out = TurboCKF._coerce_covariance(0.5, 3, "R")
        self.assertTrue(np.allclose(out, 0.5 * np.eye(3)))

    def test_coerce_covariance_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            TurboCKF._coerce_covariance(np.eye(2), 3, "R")

    def test_as_vector_rejects_bad_length(self):
        with self.assertRaises(ValueError):
            TurboCKF._as_vector(np.zeros(2), 3, "x")

    def test_stable_cholesky_adds_jitter_for_non_pd(self):
        # Exactly-singular matrix: plain cholesky fails, jitter loop recovers.
        cov = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
        chol = TurboCKF._stable_cholesky(cov)
        self.assertEqual(chol.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(chol)))
        # Lower-triangular reconstruction is close to the (jittered) input.
        self.assertTrue(np.allclose(chol @ chol.T, cov, atol=1e-3))

    def test_stable_cholesky_fallback_for_negative_definite(self):
        # Indefinite input forces the loop to exhaust and hit the 1e-6 fallback.
        cov = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=float)
        with self.assertRaises(np.linalg.LinAlgError):
            TurboCKF._stable_cholesky(cov)


class CallbackContractGuardTests(unittest.TestCase):
    """The wrapped backend model rejects malformed sigma-point batches and a
    missing dt before the user callback ever runs."""

    def test_transition_wrapper_rejects_non_2d_sigma(self):
        kf = _build()
        wrapped = kf._make_backend_model(fx_cv, expected_dim=kf.dim_x, include_dt=True)
        with self.assertRaises(ValueError):
            wrapped(np.zeros(4), kf.dt)

    def test_transition_wrapper_requires_dt_arg(self):
        kf = _build()
        wrapped = kf._make_backend_model(fx_cv, expected_dim=kf.dim_x, include_dt=True)
        with self.assertRaises(ValueError):
            wrapped(np.zeros((4, 2)))  # no dt supplied

    def test_apply_measurement_helper_rejects_bad_shape(self):
        kf = _build()
        sigma = np.zeros((4, 2))
        with self.assertRaises(ValueError):
            kf._apply_measurement(lambda x: x, sigma, ())  # returns (4,2) not (4,1)

    def test_apply_measurement_helper_happy_path(self):
        kf = _build()
        sigma = np.arange(8, dtype=float).reshape(4, 2)
        out = kf._apply_measurement(hx_pos, sigma, ())
        self.assertEqual(out.shape, (4, 1))


class AdaptiveEstimatorValidationTests(unittest.TestCase):
    def test_rejects_non_positive_diagonal_floor(self):
        kf = _build()
        with self.assertRaises(ValueError):
            kf.enable_adaptive_noise(window=5, mode="R", alpha=0.2, diagonal_floor=0.0)

    def test_rejects_non_finite_diagonal_floor(self):
        kf = _build()
        with self.assertRaises(ValueError):
            kf.enable_adaptive_noise(
                window=5, mode="R", alpha=0.2, diagonal_floor=float("inf")
            )

    def test_warmed_up_tracks_window(self):
        kf = _build()
        kf.x = np.array([0.0, 1.0], dtype=float)
        kf.enable_adaptive_noise(window=2, mode="R", alpha=0.5)
        est = kf.adaptive_noise_estimator
        self.assertFalse(est.warmed_up)
        kf.predict()
        kf.update(np.array([0.1], dtype=float))
        self.assertEqual(est.count, 1)
        self.assertFalse(est.warmed_up)
        kf.predict()
        kf.update(np.array([0.2], dtype=float))
        self.assertEqual(est.count, 2)
        self.assertTrue(est.warmed_up)

    def test_apply_adaptive_noise_skips_on_non_finite_innovation(self):
        kf = _build()
        kf.enable_adaptive_noise(window=1, mode="R", alpha=0.5)
        r_before = kf.R.copy()
        kf.y = np.array([np.nan], dtype=float)
        kf.S = np.eye(1, dtype=float)
        kf._apply_adaptive_noise()
        # Guard short-circuits: R must be untouched and estimator never stepped.
        self.assertTrue(np.allclose(kf.R, r_before))
        self.assertEqual(kf.adaptive_noise_estimator.count, 0)


class SquareRootValidationTests(unittest.TestCase):
    def test_repr_reports_downdate_counter(self):
        kf = TurboSRCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        text = repr(kf)
        self.assertIn("TurboSRCKF", text)
        self.assertIn("downdate_fallback_count", text)

    def test_predict_rejects_non_finite_dt_override(self):
        kf = TurboSRCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        with self.assertRaises(ValueError):
            kf.predict(dt=float("nan"))

    def test_transition_wrapper_rejects_non_2d_sigma(self):
        kf = TurboSRCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        wrapped = kf._make_backend_model(fx_cv, expected_dim=kf.dim_x, include_dt=True)
        with self.assertRaises(ValueError):
            wrapped(np.zeros(4), kf.dt)

    def test_transition_wrapper_requires_dt_arg(self):
        kf = TurboSRCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        wrapped = kf._make_backend_model(fx_cv, expected_dim=kf.dim_x, include_dt=True)
        with self.assertRaises(ValueError):
            wrapped(np.zeros((4, 2)))

    def test_measurement_wrapper_rejects_bad_output_shape(self):
        kf = TurboSRCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pos, fx=fx_cv)
        wrapped = kf._make_backend_model(lambda x: x, expected_dim=kf.dim_z, include_dt=False)
        with self.assertRaises(ValueError):
            wrapped(np.zeros((4, 2)))  # returns (4,2), expected (4,1)


if __name__ == "__main__":
    unittest.main()

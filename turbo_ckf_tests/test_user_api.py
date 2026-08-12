"""User-facing ergonomics: coercing state setters, NaN-measurement guard,
and dt propagation to the backend."""

import unittest

import numpy as np

from turbo_ckf import TurboCKF, TurboSRCKF


def fx(sp, dt):
    out = np.empty_like(sp)
    out[:, 0] = sp[:, 0] + dt * sp[:, 1]
    out[:, 1] = sp[:, 1]
    return out


def hx(sp):
    return sp[:, 0:1]


def make_ckf(**kwargs):
    return TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx, **kwargs)


def make_srckf(**kwargs):
    return TurboSRCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx, **kwargs)


class StateCoercionTests(unittest.TestCase):
    def test_x_accepts_list(self):
        kf = make_ckf()
        kf.x = [1.0, 2.0]
        self.assertEqual(kf.x.dtype, np.float64)
        np.testing.assert_allclose(kf.x, [1.0, 2.0])
        kf.predict()  # must not raise inside the backend push

    def test_x_accepts_int_array(self):
        kf = make_ckf()
        kf.x = np.array([1, 2])
        self.assertEqual(kf.x.dtype, np.float64)
        kf.predict()

    def test_x_accepts_column_and_row_vectors(self):
        kf = make_ckf()
        kf.x = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(kf.x, [1.0, 2.0])
        kf.x = np.array([[3.0, 4.0]])
        np.testing.assert_allclose(kf.x, [3.0, 4.0])

    def test_x_wrong_size_raises_at_assignment(self):
        kf = make_ckf()
        with self.assertRaisesRegex(ValueError, "x must be a length-2"):
            kf.x = [1.0, 2.0, 3.0]

    def test_x_non_finite_raises(self):
        kf = make_ckf()
        with self.assertRaisesRegex(ValueError, "finite"):
            kf.x = [np.nan, 0.0]

    def test_x_non_numeric_raises_typeerror(self):
        kf = make_ckf()
        with self.assertRaises((TypeError, ValueError)):
            kf.x = ["a", "b"]

    def test_covariances_accept_scalar(self):
        kf = make_ckf()
        kf.R = 0.25
        np.testing.assert_allclose(kf.R, [[0.25]])
        kf.Q = 2.0
        np.testing.assert_allclose(kf.Q, 2.0 * np.eye(2))
        kf.predict()
        kf.update(np.array([0.1]))

    def test_covariances_accept_diagonal(self):
        kf = make_ckf()
        kf.Q = [1e-3, 1e-2]
        np.testing.assert_allclose(kf.Q, np.diag([1e-3, 1e-2]))

    def test_covariance_wrong_shape_raises_at_assignment(self):
        kf = make_ckf()
        with self.assertRaisesRegex(ValueError, "P must be"):
            kf.P = np.eye(3)

    def test_covariance_non_finite_raises(self):
        kf = make_ckf()
        with self.assertRaisesRegex(ValueError, "finite"):
            kf.P = np.full((2, 2), np.nan)

    def test_int_covariance_accepted(self):
        kf = make_ckf()
        kf.P = np.eye(2, dtype=int)
        self.assertEqual(kf.P.dtype, np.float64)
        kf.predict()

    def test_srckf_coercion_mirror(self):
        kf = make_srckf()
        kf.x = [1, 2]
        kf.R = 0.25
        kf.Q = [1e-3, 1e-3]
        kf.predict()
        kf.update(np.array([0.9]))
        with self.assertRaisesRegex(ValueError, "x must be a length-2"):
            kf.x = [1.0]


class NanMeasurementGuardTests(unittest.TestCase):
    def test_nan_z_raises_with_hint(self):
        kf = make_ckf()
        kf.predict()
        with self.assertRaisesRegex(ValueError, "z=None"):
            kf.update(np.array([np.nan]))
        # State untouched by the rejected update.
        self.assertTrue(np.all(np.isfinite(kf.x)))
        kf.update(np.array([0.2]))  # filter still healthy
        self.assertTrue(np.all(np.isfinite(kf.x)))

    def test_inf_z_raises(self):
        kf = make_ckf()
        kf.predict()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            kf.update(np.array([np.inf]))

    def test_z_none_still_skips_cleanly(self):
        kf = make_ckf()
        kf.predict()
        kf.update(None)
        self.assertTrue(np.all(np.isnan(kf.z)))
        self.assertTrue(np.all(np.isfinite(kf.x)))

    def test_srckf_nan_z_raises(self):
        kf = make_srckf()
        kf.predict()
        with self.assertRaisesRegex(ValueError, "z=None"):
            kf.update(np.array([np.nan]))
        self.assertTrue(np.all(np.isfinite(kf.x)))


class DtPropagationTests(unittest.TestCase):
    def test_standard_model_honors_dt_change(self):
        kf = make_ckf()
        kf.x = [0.0, 1.0]
        kf.dt = 1.0
        kf.predict_standard_model("constant_velocity")
        self.assertAlmostEqual(kf.x[0], 1.0)

    def test_standard_model_ckf_honors_dt_change(self):
        kf = make_ckf()
        kf.x = [0.0, 1.0]
        kf.dt = 0.5
        kf.predict_standard_model_ckf("constant_velocity")
        self.assertAlmostEqual(kf.x[0], 0.5, places=6)

    def test_predict_default_uses_current_dt(self):
        kf = make_ckf()
        kf.x = [0.0, 1.0]
        kf.dt = 2.0
        kf.predict()
        self.assertAlmostEqual(kf.x[0], 2.0, places=6)

    def test_dt_must_be_finite(self):
        kf = make_ckf()
        with self.assertRaisesRegex(ValueError, "dt must be finite"):
            kf.dt = np.nan

    def test_srckf_dt_property(self):
        kf = make_srckf()
        kf.x = [0.0, 1.0]
        kf.dt = 1.0
        kf.predict()
        self.assertAlmostEqual(kf.x[0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

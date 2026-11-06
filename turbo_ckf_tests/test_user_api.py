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

    def test_covariance_wrong_diagonal_length_raises(self):
        kf = make_ckf()
        with self.assertRaisesRegex(ValueError, "diagonal"):
            kf.Q = [1e-3]

    def test_covariance_non_numeric_raises_typeerror(self):
        kf = make_ckf()
        with self.assertRaisesRegex(TypeError, "Q must be"):
            kf.Q = {"not": "a matrix"}

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


def hx_identity(sp):
    return sp.copy()


class StridedMeasurementTests(unittest.TestCase):
    """A 1-D measurement that is a strided view, such as a column Z[:, k] of a
    (dim_z, N) array, used to raise TypeError "The given array is not
    contiguous". It must give the same result as a contiguous copy."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.Z = rng.normal(size=(2, 12))  # column k is one measurement
        self.assertFalse(self.Z[:, 0].flags.c_contiguous)

    def _pair(self, cls):
        a = cls(dim_x=2, dim_z=2, dt=0.1, hx=hx_identity, fx=fx)
        b = cls(dim_x=2, dim_z=2, dt=0.1, hx=hx_identity, fx=fx)
        return a, b

    def test_ckf_update_strided(self):
        a, b = self._pair(TurboCKF)
        for k in range(self.Z.shape[1]):
            a.predict()
            a.update(self.Z[:, k])
            b.predict()
            b.update(np.ascontiguousarray(self.Z[:, k]))
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.P, b.P)

    def test_srckf_update_strided(self):
        a, b = self._pair(TurboSRCKF)
        for k in range(self.Z.shape[1]):
            a.predict()
            a.update(self.Z[:, k])
            b.predict()
            b.update(np.ascontiguousarray(self.Z[:, k]))
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.P, b.P)

    def test_run_list_of_strided_columns(self):
        a, b = self._pair(TurboCKF)
        ra = a.run([self.Z[:, k] for k in range(self.Z.shape[1])])
        rb = b.run(np.ascontiguousarray(self.Z.T))
        np.testing.assert_array_equal(ra.xs, rb.xs)
        np.testing.assert_array_equal(ra.Ps, rb.Ps)

    def test_update_paper_ahrs_strided(self):
        zs = np.array(
            [[0.1, 0.2], [-0.05, 0.0], [9.8, 9.7], [40.0, 41.0], [2.0, 1.5], [30.0, 29.0]]
        )
        z = zs[:, 0]
        self.assertFalse(z.flags.c_contiguous)
        kfs = []
        for meas in (z, np.ascontiguousarray(z)):
            kf = TurboCKF(4, 6, 0.01, hx=lambda s: np.zeros((8, 6)), fx=lambda s, dt: s)
            kf.x = [1.0, 0.0, 0.0, 0.0]
            kf.P = 1e-2 * np.eye(4)
            kf.update_paper_ahrs(meas, sigma_acc2=1e-2, sigma_mag2=1e-2)
            kfs.append(kf)
        np.testing.assert_array_equal(kfs[0].x, kfs[1].x)
        np.testing.assert_array_equal(kfs[0].P, kfs[1].P)


if __name__ == "__main__":
    unittest.main()

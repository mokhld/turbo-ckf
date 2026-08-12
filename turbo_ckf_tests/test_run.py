"""TurboCKF.run / TurboSRCKF.run — whole-sequence predict+update in one call."""

import unittest

import numpy as np

from turbo_ckf import FilterRun, TurboCKF, TurboSRCKF


def fx(sp, dt):
    out = np.empty_like(sp)
    out[:, 0] = sp[:, 0] + dt * sp[:, 1]
    out[:, 1] = sp[:, 1]
    return out


def hx(sp):
    return sp[:, 0:1]


def make_ckf(cls=TurboCKF):
    kf = cls(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
    kf.x = [0.0, 1.0]
    kf.P = np.eye(2)
    kf.Q = 1e-3 * np.eye(2)
    kf.R = 0.25
    return kf


def make_zs(n=50, seed=7):
    rng = np.random.default_rng(seed)
    true_pos = np.cumsum(np.full(n, 0.1 * 1.5))
    return (true_pos + rng.normal(0.0, 0.5, size=n)).reshape(n, 1)


class RunEquivalenceTests(unittest.TestCase):
    def test_run_matches_manual_loop(self):
        zs = make_zs()
        kf_a = make_ckf()
        kf_b = make_ckf()

        result = kf_a.run(zs)

        xs_manual = []
        for z in zs:
            kf_b.predict()
            kf_b.update(z)
            xs_manual.append(kf_b.x.copy())

        self.assertIsInstance(result, FilterRun)
        np.testing.assert_allclose(result.xs, np.array(xs_manual))
        np.testing.assert_allclose(kf_a.x, kf_b.x)
        np.testing.assert_allclose(kf_a.P, kf_b.P)
        self.assertEqual(result.xs.shape, (len(zs), 2))
        self.assertEqual(result.Ps.shape, (len(zs), 2, 2))
        self.assertEqual(result.log_likelihoods.shape, (len(zs),))
        self.assertFalse(result.missing.any())
        # Final state mirrors the last row.
        np.testing.assert_allclose(kf_a.x, result.xs[-1])

    def test_srckf_run_matches_manual_loop(self):
        zs = make_zs()
        kf_a = make_ckf(TurboSRCKF)
        kf_b = make_ckf(TurboSRCKF)
        result = kf_a.run(zs)
        for z in zs:
            kf_b.predict()
            kf_b.update(z)
        np.testing.assert_allclose(kf_a.x, kf_b.x)
        np.testing.assert_allclose(result.xs[-1], kf_b.x)

    def test_one_dimensional_zs_for_scalar_measurements(self):
        zs = make_zs()
        a = make_ckf().run(zs)
        b = make_ckf().run(zs.reshape(-1))
        np.testing.assert_allclose(a.xs, b.xs)


class MissingMeasurementTests(unittest.TestCase):
    def test_none_entries_skip_update(self):
        zs = [np.array([0.2]), None, np.array([0.5]), None]
        kf = make_ckf()
        result = kf.run(zs)
        np.testing.assert_array_equal(result.missing, [False, True, False, True])
        self.assertTrue(np.isnan(result.log_likelihoods[1]))
        self.assertTrue(np.isnan(result.nis[1]))
        self.assertFalse(np.isnan(result.log_likelihoods[2]))
        self.assertTrue(np.all(np.isfinite(result.xs)))
        # A skipped step's posterior equals its prior.
        np.testing.assert_allclose(result.xs[1], result.x_priors[1])

    def test_scalar_entries_in_list(self):
        kf = make_ckf()
        result = kf.run([0.2, 0.3, None, 0.5])
        self.assertEqual(result.xs.shape, (4, 2))
        self.assertTrue(result.missing[2])

    def test_nan_rows_raise_by_default(self):
        zs = make_zs(5)
        zs[2, 0] = np.nan
        kf = make_ckf()
        x_before = kf.x.copy()
        with self.assertRaisesRegex(ValueError, "nan_means_missing"):
            kf.run(zs)
        # Validation is up-front: the filter never advanced.
        np.testing.assert_allclose(kf.x, x_before)

    def test_nan_means_missing_skips_all_nan_rows(self):
        zs = make_zs(5)
        zs[2, 0] = np.nan
        result = make_ckf().run(zs, nan_means_missing=True)
        np.testing.assert_array_equal(result.missing, [False, False, True, False, False])
        self.assertTrue(np.all(np.isfinite(result.xs)))

    def test_partial_nan_rows_always_raise(self):
        kf = TurboCKF(dim_x=2, dim_z=2, dt=0.1, hx=lambda sp: sp, fx=fx)
        zs = np.zeros((3, 2))
        zs[1] = [np.nan, 1.0]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            kf.run(zs, nan_means_missing=True)


class RunOptionsTests(unittest.TestCase):
    def test_per_step_dts(self):
        zs = make_zs(4)
        dts = np.array([0.1, 0.2, 0.3, 0.4])
        kf_a = make_ckf()
        kf_b = make_ckf()
        result = kf_a.run(zs, dts=dts)
        for z, dt in zip(zs, dts):
            kf_b.predict(dt=dt)
            kf_b.update(z)
        np.testing.assert_allclose(result.xs[-1], kf_b.x)

    def test_shared_scalar_R(self):
        zs = make_zs(10)
        a = make_ckf().run(zs, Rs=0.5)
        kf_b = make_ckf()
        kf_b.R = 0.5
        b = kf_b.run(zs)
        np.testing.assert_allclose(a.xs, b.xs)

    def test_per_step_Rs(self):
        zs = make_zs(3)
        rs = np.stack([0.1 * np.eye(1), 0.2 * np.eye(1), 0.3 * np.eye(1)])
        kf_a = make_ckf()
        kf_b = make_ckf()
        result = kf_a.run(zs, Rs=rs)
        for z, r in zip(zs, rs):
            kf_b.predict()
            kf_b.update(z, R=r)
        self.assertEqual(result.xs.shape, (3, 2))
        np.testing.assert_allclose(result.xs[-1], kf_b.x)

    def test_adaptive_noise_composes_with_run(self):
        zs = make_zs(60)
        kf = make_ckf()
        kf.enable_adaptive_noise(window=10)
        result = kf.run(zs)
        self.assertTrue(np.all(np.isfinite(result.xs)))
        self.assertGreaterEqual(kf.adaptive_noise_estimator.count, 60)


class RunValidationTests(unittest.TestCase):
    def test_empty_zs_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            make_ckf().run([])

    def test_bad_shape_raises_before_any_step(self):
        kf = make_ckf()
        x_before = kf.x.copy()
        with self.assertRaises(ValueError):
            kf.run([np.array([0.1]), np.array([0.1, 0.2])])
        np.testing.assert_allclose(kf.x, x_before)

    def test_bad_dts_length_raises(self):
        with self.assertRaisesRegex(ValueError, "dts"):
            make_ckf().run(make_zs(4), dts=np.array([0.1, 0.2]))

    def test_bad_Rs_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "Rs"):
            make_ckf().run(make_zs(4), Rs=np.zeros((2, 1, 1)))


if __name__ == "__main__":
    unittest.main()

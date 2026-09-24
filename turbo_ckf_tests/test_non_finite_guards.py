"""NaN/inf guards on the batch paths and on callback outputs (REVIEW A5).

One non-finite value used to enter the filter state without an error:
``batch_filter`` returned NaN for every row after a NaN observation,
``batch_parallel_step`` reported status 0 next to a NaN state, and an
``fx``/``hx`` returning NaN corrupted ``x`` and ``P`` until the next call
failed with an unrelated "unable to compute stable Cholesky factor".
"""

from __future__ import annotations

import unittest

import numpy as np

from turbo_ckf import TurboCKF, TurboSRCKF, _rust, batch_filter, batch_parallel_step

F = np.array([[1.0, 0.1], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = 0.01 * np.eye(2)
R = np.array([[0.25]])


def fx_cv(sigmas, dt):
    out = sigmas.copy()
    out[:, 0] = sigmas[:, 0] + dt * sigmas[:, 1]
    return out


def hx_pos(sigmas):
    return sigmas[:, :1]


def fx_nan_row1(sigmas, dt):
    out = fx_cv(sigmas, dt)
    out[1, 0] = np.nan
    return out


def hx_inf_row2(sigmas):
    out = hx_pos(sigmas).copy()
    out[2, 0] = np.inf
    return out


class BatchFilterNonFiniteTests(unittest.TestCase):
    @staticmethod
    def _zs() -> np.ndarray:
        return np.linspace(0.0, 1.0, 10).reshape(-1, 1)

    def test_nan_measurement_raises_with_row_and_run_hint(self):
        zs = self._zs()
        zs[3, 0] = np.nan
        with self.assertRaisesRegex(ValueError, r"^zs\[3\] contains non-finite") as ctx:
            batch_filter(np.zeros(2), np.eye(2), zs, F=F, H=H, Q=Q, R=R)
        message = str(ctx.exception)
        self.assertIn("does not support missing measurements", message)
        self.assertIn("TurboCKF.run(..., nan_means_missing=True)", message)

    def test_inf_measurement_raises(self):
        zs = self._zs()
        zs[7, 0] = -np.inf
        with self.assertRaisesRegex(ValueError, r"^zs\[7\] contains non-finite"):
            batch_filter(np.zeros(2), np.eye(2), zs, F=F, H=H, Q=Q, R=R)

    def test_non_finite_x0_and_p0_raise(self):
        cases = {
            "x0": (np.array([np.nan, 0.0]), np.eye(2)),
            "P0": (np.zeros(2), np.array([[1.0, 0.0], [0.0, np.inf]])),
        }
        for name, (x0, p0) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"^{name} contains non-finite"):
                    batch_filter(x0, p0, self._zs(), F=F, H=H, Q=Q, R=R)

    def test_non_finite_model_matrix_raises_with_step(self):
        n = 10
        per_step = {
            "F": np.tile(F, (n, 1, 1)),
            "H": np.tile(H, (n, 1, 1)),
            "Q": np.tile(Q, (n, 1, 1)),
            "R": np.tile(R, (n, 1, 1)),
        }
        for name in per_step:
            with self.subTest(name=name):
                mats = {k: v.copy() for k, v in per_step.items()}
                mats[name][4, 0, 0] = np.nan
                with self.assertRaisesRegex(
                    ValueError, rf"^{name} contains non-finite values \(NaN or inf\) at step 4"
                ):
                    batch_filter(np.zeros(2), np.eye(2), self._zs(), **mats)


class BatchParallelStepNonFiniteTests(unittest.TestCase):
    @staticmethod
    def _bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, -1.0]])
        Ps = np.stack([np.eye(2), 2.0 * np.eye(2), np.array([[0.5, 0.1], [0.1, 0.4]])])
        zs = np.array([[0.1], [1.2], [1.8]])
        return xs, Ps, zs

    def test_non_finite_z_gives_status_3_and_predict_step_output(self):
        xs, Ps, zs = self._bank()
        ref_xs, ref_Ps, ref_lls, _ = batch_parallel_step(xs, Ps, zs, F=F, H=H, Q=Q, R=R)
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                zs_bad = zs.copy()
                zs_bad[1, 0] = bad
                new_xs, new_Ps, lls, status = batch_parallel_step(
                    xs, Ps, zs_bad, F=F, H=H, Q=Q, R=R
                )
                np.testing.assert_array_equal(status, [0, 3, 0])
                np.testing.assert_allclose(new_xs[1], F @ xs[1], rtol=1e-12)
                np.testing.assert_allclose(new_Ps[1], F @ Ps[1] @ F.T + Q, rtol=1e-12)
                self.assertEqual(lls[1], -np.inf)
                # One bad filter does not affect the rest of the bank.
                for i in (0, 2):
                    np.testing.assert_array_equal(new_xs[i], ref_xs[i])
                    np.testing.assert_array_equal(new_Ps[i], ref_Ps[i])
                    self.assertEqual(lls[i], ref_lls[i])

    def test_non_finite_x_or_p_returns_inputs_unchanged(self):
        xs, Ps, zs = self._bank()
        xs[0, 1] = np.nan
        Ps[2, 0, 1] = np.inf
        new_xs, new_Ps, lls, status = batch_parallel_step(xs, Ps, zs, F=F, H=H, Q=Q, R=R)
        np.testing.assert_array_equal(status, [3, 0, 3])
        for i in (0, 2):
            np.testing.assert_array_equal(new_xs[i], xs[i])
            np.testing.assert_array_equal(new_Ps[i], Ps[i])
            self.assertTrue(np.isnan(lls[i]))
        self.assertTrue(np.all(np.isfinite(new_xs[1])))
        self.assertTrue(np.isfinite(lls[1]))

    def test_non_finite_shared_matrix_raises(self):
        xs, Ps, zs = self._bank()
        shared = {"F": F, "H": H, "Q": Q, "R": R}
        for name in shared:
            with self.subTest(name=name):
                mats = {k: v.copy() for k, v in shared.items()}
                mats[name][0, 0] = np.nan
                with self.assertRaisesRegex(ValueError, rf"^{name} contains non-finite"):
                    batch_parallel_step(xs, Ps, zs, **mats)


class CallbackNonFiniteTests(unittest.TestCase):
    """A NaN/inf from fx or hx raises, names the callback, and leaves the
    filter as it was."""

    @staticmethod
    def _make(cls):
        kf = cls(dim_x=2, dim_z=1, dt=0.1, fx=fx_cv, hx=hx_pos)
        kf.x = [0.5, 1.0]
        kf.P = [[1.0, 0.1], [0.1, 0.5]]
        kf.Q = 1e-3
        kf.R = 0.25
        kf.predict()
        kf.update(np.array([0.6]))
        return kf

    def test_fx_non_finite_raises_and_keeps_state(self):
        for cls in (TurboCKF, TurboSRCKF):
            with self.subTest(cls=cls.__name__):
                kf = self._make(cls)
                x_before, P_before = kf.x.copy(), kf.P.copy()
                with self.assertRaisesRegex(
                    ValueError, r"^fx returned non-finite values \(NaN or inf\) in output row 1"
                ):
                    kf.predict(fx=fx_nan_row1)
                np.testing.assert_array_equal(kf.x, x_before)
                np.testing.assert_array_equal(kf.P, P_before)
                kf.predict()
                kf.update(np.array([0.7]))
                self.assertTrue(np.all(np.isfinite(kf.x)))

    def test_hx_non_finite_raises_and_keeps_state(self):
        for cls in (TurboCKF, TurboSRCKF):
            with self.subTest(cls=cls.__name__):
                kf = self._make(cls)
                kf.predict()
                x_before, P_before = kf.x.copy(), kf.P.copy()
                with self.assertRaisesRegex(
                    ValueError, r"^hx returned non-finite values \(NaN or inf\) in output row 2"
                ):
                    kf.update(np.array([0.7]), hx=hx_inf_row2)
                np.testing.assert_array_equal(kf.x, x_before)
                np.testing.assert_array_equal(kf.P, P_before)
                kf.update(np.array([0.7]))
                self.assertTrue(np.all(np.isfinite(kf.x)))

    def test_backend_state_unchanged_after_bad_callback(self):
        # Checks the Rust objects directly, independent of how the Python
        # wrapper syncs state: x and P are assigned only after the callback
        # output passes the check.
        x = np.array([0.5, 1.0])
        P = np.array([[1.0, 0.1], [0.1, 0.5]])
        q = 1e-3 * np.eye(2)
        r = np.array([[0.25]])
        for backend_cls in (_rust.CubatureKalmanFilter, _rust.SquareRootCubatureKalmanFilter):
            with self.subTest(cls=backend_cls.__name__):
                backend = backend_cls(2, 1, 0.1)
                backend.set_state(x, P, q, r)
                before = backend.snapshot()
                with self.assertRaisesRegex(ValueError, r"^fx returned non-finite"):
                    backend.predict_custom(fx_nan_row1, 0.1, ())
                with self.assertRaisesRegex(ValueError, r"^hx returned non-finite"):
                    backend.update(np.array([0.6]), hx_inf_row2, r, ())
                after = backend.snapshot()
                for key in ("x", "P", "x_prior", "P_prior", "x_post", "P_post"):
                    np.testing.assert_array_equal(after[key], before[key], err_msg=key)


if __name__ == "__main__":
    unittest.main()

"""Linear Kalman batch_filter correctness tests.

The headline correctness check is parity vs a sequential predict/update
loop using TurboCKF's existing per-step API: the batch routine must
agree to ~1e-10. Plus shape coverage, constant-vs-tiled equivalence,
log-likelihood sanity, broadcast defaults, composability with
`rts_smooth`, and shape-rejection paths.
"""

from __future__ import annotations

import unittest

import numpy as np

from turbo_ckf import TurboCKF, batch_filter, rts_smooth


def _cv_matrices(dt: float, q_var: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    g = np.array([[0.5 * dt * dt], [dt]], dtype=float)
    q = q_var * (g @ g.T)
    return f, q


def fx_cv(x: np.ndarray, dt: float) -> np.ndarray:
    out = x.copy()
    out[:, 0] = x[:, 0] + dt * x[:, 1]
    return out


def hx_pos(x: np.ndarray) -> np.ndarray:
    return x[:, :1]


def _simulate_cv(seed: int, n: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_var = 0.5
    r_var = 0.25
    f_mat, _ = _cv_matrices(dt, q_var)
    truth = np.zeros((n, 2))
    truth[0] = np.array([0.0, 1.0])
    for k in range(1, n):
        a = rng.normal(0.0, np.sqrt(q_var))
        truth[k] = f_mat @ truth[k - 1] + np.array([0.5 * dt * dt * a, dt * a])
    obs = (truth[:, 0] + rng.normal(0.0, np.sqrt(r_var), size=n)).reshape(-1, 1)
    return truth, obs


class BatchFilterParityTests(unittest.TestCase):
    """The marquee test: batch_filter must match a sequential per-step loop."""

    def test_batch_matches_sequential_predict_update(self):
        truth, zs = _simulate_cv(seed=0, n=300, dt=0.1)
        dt = 0.1
        f_mat, q_mat = _cv_matrices(dt, q_var=0.5)
        h_mat = np.array([[1.0, 0.0]], dtype=float)
        r_mat = np.array([[0.25]], dtype=float)
        x0 = np.array([0.0, 0.5], dtype=float)
        p0 = np.eye(2)

        # Sequential reference via TurboCKF's per-step API. Use a
        # vectorized linear hx so the existing update() matches the
        # linear KF math the batch routine implements.
        def hx_linear(x: np.ndarray) -> np.ndarray:
            return (h_mat @ x.T).T

        kf = TurboCKF(dim_x=2, dim_z=1, dt=dt, hx=hx_linear, fx=fx_cv)
        kf.x = x0.copy()
        kf.P = p0.copy()
        kf.Q = q_mat
        kf.R = r_mat
        ref_xs = np.zeros((zs.shape[0], 2))
        ref_Ps = np.zeros((zs.shape[0], 2, 2))
        ref_lls = np.zeros(zs.shape[0])
        for k in range(zs.shape[0]):
            kf.predict_linear_model(f_mat)
            kf.update(zs[k])
            ref_xs[k] = kf.x
            ref_Ps[k] = kf.P
            ref_lls[k] = kf.log_likelihood

        # Batch run.
        xs, Ps, lls = batch_filter(x0, p0, zs, F=f_mat, H=h_mat, Q=q_mat, R=r_mat)

        # State + covariance parity: tight numerical agreement.
        self.assertTrue(np.allclose(xs, ref_xs, atol=1e-9))
        self.assertTrue(np.allclose(Ps, ref_Ps, atol=1e-9))
        # Per-step log-likelihoods agree.
        self.assertTrue(np.allclose(lls, ref_lls, atol=1e-9))


class BatchFilterRtsCompositionTests(unittest.TestCase):
    """batch_filter -> rts_smooth must drop position RMSE by >= 30% vs the
    forward filter, just like the sequential pipeline."""

    def test_batch_then_smooth_cuts_rmse(self):
        truth, zs = _simulate_cv(seed=7, n=400, dt=0.1)
        dt = 0.1
        f_mat, q_mat = _cv_matrices(dt, q_var=0.5)
        h_mat = np.array([[1.0, 0.0]], dtype=float)
        r_mat = np.array([[0.25]], dtype=float)
        x0 = np.array([0.0, 0.5], dtype=float)
        p0 = np.eye(2)

        xs, Ps, _ = batch_filter(x0, p0, zs, F=f_mat, H=h_mat, Q=q_mat, R=r_mat)
        xs_s, Ps_s = rts_smooth(xs, Ps, np.tile(f_mat, (400, 1, 1)), np.tile(q_mat, (400, 1, 1)))

        rmse_f = float(np.sqrt(np.mean((xs[:, 0] - truth[:, 0]) ** 2)))
        rmse_s = float(np.sqrt(np.mean((xs_s[:, 0] - truth[:, 0]) ** 2)))
        self.assertLess(rmse_s, 0.7 * rmse_f,
                        msg=f"smoothed RMSE {rmse_s:.4f} not <= 0.7 * filtered RMSE {rmse_f:.4f}")


class BatchFilterBroadcastTests(unittest.TestCase):
    def test_constant_matches_tiled_per_step(self):
        truth, zs = _simulate_cv(seed=3, n=80, dt=0.1)
        f_mat, q_mat = _cv_matrices(0.1, q_var=0.2)
        h_mat = np.array([[1.0, 0.0]], dtype=float)
        r_mat = np.array([[0.3]], dtype=float)
        x0 = np.array([0.0, 1.0])
        p0 = np.eye(2)

        xs_a, Ps_a, lls_a = batch_filter(x0, p0, zs, F=f_mat, H=h_mat, Q=q_mat, R=r_mat)
        n = zs.shape[0]
        xs_b, Ps_b, lls_b = batch_filter(
            x0, p0, zs,
            F=np.tile(f_mat, (n, 1, 1)),
            H=np.tile(h_mat, (n, 1, 1)),
            Q=np.tile(q_mat, (n, 1, 1)),
            R=np.tile(r_mat, (n, 1, 1)),
        )
        self.assertTrue(np.allclose(xs_a, xs_b))
        self.assertTrue(np.allclose(Ps_a, Ps_b))
        self.assertTrue(np.allclose(lls_a, lls_b))

    def test_defaults_q_zero_r_identity(self):
        zs = np.array([[1.0], [2.0], [3.0]])
        f = np.eye(2)
        h = np.array([[1.0, 0.0]])
        x0 = np.zeros(2)
        p0 = np.eye(2)
        # Should not raise; R defaults to I(1), Q defaults to 0.
        xs, Ps, lls = batch_filter(x0, p0, zs, F=f, H=h)
        self.assertEqual(xs.shape, (3, 2))
        self.assertEqual(Ps.shape, (3, 2, 2))
        self.assertEqual(lls.shape, (3,))
        self.assertTrue(np.all(np.isfinite(xs)))

    def test_per_step_f_handled(self):
        """Drifting F across the trajectory must run cleanly and produce
        finite, PSD covariances."""
        _, zs = _simulate_cv(seed=11, n=100, dt=0.1)
        dt = 0.1
        f_a, q_mat = _cv_matrices(dt, q_var=0.2)
        f_b = np.array([[1.0, 0.05], [0.0, 1.0]], dtype=float)
        fs = np.concatenate([np.tile(f_a, (50, 1, 1)), np.tile(f_b, (50, 1, 1))])
        h_mat = np.array([[1.0, 0.0]])
        r_mat = np.array([[0.3]])
        xs, Ps, _ = batch_filter(np.zeros(2), np.eye(2), zs, F=fs, H=h_mat, Q=q_mat, R=r_mat)
        self.assertTrue(np.all(np.isfinite(xs)))
        self.assertTrue(np.all(np.isfinite(Ps)))
        for k in range(Ps.shape[0]):
            self.assertLess(float(np.max(np.abs(Ps[k] - Ps[k].T))), 1e-10)
            eig = float(np.min(np.linalg.eigvalsh(0.5 * (Ps[k] + Ps[k].T))))
            self.assertGreater(eig, -1e-10)


class BatchFilterLogLikelihoodTests(unittest.TestCase):
    def test_log_likelihood_is_finite_for_well_posed_problem(self):
        truth, zs = _simulate_cv(seed=2, n=50, dt=0.1)
        f_mat, q_mat = _cv_matrices(0.1, q_var=0.2)
        h_mat = np.array([[1.0, 0.0]])
        r_mat = np.array([[0.25]])
        _, _, lls = batch_filter(np.zeros(2), np.eye(2), zs, F=f_mat, H=h_mat, Q=q_mat, R=r_mat)
        self.assertTrue(np.all(np.isfinite(lls)))
        # For a 1-D measurement around order 1, log-likelihood per step is
        # typically negative but not pathologically so (no -1e10 nonsense).
        self.assertGreater(float(np.mean(lls)), -10.0)


class BatchFilterShapeRejectionTests(unittest.TestCase):
    def test_bad_shapes_raise(self):
        x0 = np.zeros(2)
        p0 = np.eye(2)
        zs = np.zeros((5, 1))
        f = np.eye(2)
        h = np.array([[1.0, 0.0]])

        with self.assertRaises(ValueError):
            batch_filter(x0, p0, np.zeros(5), F=f, H=h)  # 1D zs
        with self.assertRaises(ValueError):
            batch_filter(x0, np.zeros((3, 3)), zs, F=f, H=h)  # bad P0
        with self.assertRaises(ValueError):
            batch_filter(x0, p0, zs, F=np.eye(3), H=h)  # bad F
        with self.assertRaises(ValueError):
            batch_filter(x0, p0, zs, F=f, H=np.array([[1.0, 0.0, 0.0]]))  # bad H
        with self.assertRaises(ValueError):
            batch_filter(x0, p0, zs, F=np.tile(f, (4, 1, 1)), H=h)  # bad per-step len
        with self.assertRaises(ValueError):
            batch_filter(x0, p0, np.zeros((0, 1)), F=f, H=h)  # zero observations

    def test_classmethod_form_also_works(self):
        truth, zs = _simulate_cv(seed=5, n=30, dt=0.1)
        f_mat, q_mat = _cv_matrices(0.1, q_var=0.2)
        h_mat = np.array([[1.0, 0.0]])
        r_mat = np.array([[0.25]])
        xs_a, _, _ = batch_filter(np.zeros(2), np.eye(2), zs, F=f_mat, H=h_mat, Q=q_mat, R=r_mat)
        xs_b, _, _ = TurboCKF.batch_filter(np.zeros(2), np.eye(2), zs, F=f_mat, H=h_mat, Q=q_mat, R=r_mat)
        self.assertTrue(np.allclose(xs_a, xs_b))


if __name__ == "__main__":
    unittest.main()

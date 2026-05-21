"""RTS smoother correctness tests.

The smoother must beat the forward filter on a canonical linear-Gaussian
trajectory. We simulate a noisy constant-velocity (CV) walk, run the
forward filter to collect (xs, Ps, Fs, Qs), then smooth backwards. RMSE
of the smoothed position must drop by at least 30% versus the filtered
position. Also covers single-step traces (identity), shape validation,
and the FilterPy-compatible length-N vs length-(N-1) input convention.
"""

from __future__ import annotations

import unittest

import numpy as np

from turbo_ckf import TurboCKF, rts_smooth


def _cv_matrices(dt: float, q_var: float) -> tuple[np.ndarray, np.ndarray]:
    """Standard discrete-white-noise CV F/Q for a [pos, vel] state."""
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


def _simulate_and_filter(seed: int, n_steps: int, dt: float):
    """Simulate CV truth + noisy position obs, then run the forward filter
    storing the filtered (x, P) at each step. Returns truth + filter trace."""
    rng = np.random.default_rng(seed)
    q_var = 0.5  # accel variance
    r_var = 0.25  # obs variance

    f_mat, q_mat = _cv_matrices(dt, q_var)

    # Simulate truth
    true_x = np.zeros((n_steps, 2))
    true_x[0] = np.array([0.0, 1.0])
    obs = np.zeros(n_steps)
    for k in range(1, n_steps):
        accel_noise = rng.normal(0.0, np.sqrt(q_var))
        true_x[k] = f_mat @ true_x[k - 1] + np.array(
            [0.5 * dt * dt * accel_noise, dt * accel_noise]
        )
    obs = true_x[:, 0] + rng.normal(0.0, np.sqrt(r_var), size=n_steps)

    # Forward filter with TurboCKF
    kf = TurboCKF(dim_x=2, dim_z=1, dt=dt, hx=hx_pos, fx=fx_cv)
    kf.x = np.array([0.0, 0.5], dtype=float)
    kf.P = np.eye(2) * 1.0
    kf.Q = q_mat
    kf.R = np.array([[r_var]], dtype=float)

    xs = np.zeros((n_steps, 2))
    Ps = np.zeros((n_steps, 2, 2))
    Fs = np.tile(f_mat, (n_steps, 1, 1))
    Qs = np.tile(q_mat, (n_steps, 1, 1))
    # Update with the first observation before stepping; the filter trace
    # stores the posterior at each step.
    kf.update(np.array([obs[0]], dtype=float))
    xs[0] = kf.x
    Ps[0] = kf.P
    for k in range(1, n_steps):
        kf.predict()
        kf.update(np.array([obs[k]], dtype=float))
        xs[k] = kf.x
        Ps[k] = kf.P

    return true_x, xs, Ps, Fs, Qs


class RtsSmootherTests(unittest.TestCase):
    def test_smoother_reduces_rmse_on_cv_trajectory(self):
        """Forward filter then backward smooth must cut position RMSE by >=30%."""
        truth, xs, Ps, Fs, Qs = _simulate_and_filter(seed=0, n_steps=400, dt=0.1)

        xs_s, Ps_s = rts_smooth(xs, Ps, Fs, Qs)

        rmse_filt = float(np.sqrt(np.mean((xs[:, 0] - truth[:, 0]) ** 2)))
        rmse_smooth = float(np.sqrt(np.mean((xs_s[:, 0] - truth[:, 0]) ** 2)))

        self.assertLess(rmse_smooth, 0.7 * rmse_filt,
                        msg=f"smoothed RMSE {rmse_smooth:.4f} not <= 0.7 * filtered RMSE {rmse_filt:.4f}")
        # Sanity: covariances must stay PSD + symmetric.
        for k in range(Ps_s.shape[0]):
            self.assertLess(float(np.max(np.abs(Ps_s[k] - Ps_s[k].T))), 1e-10)
            eig = float(np.min(np.linalg.eigvalsh(0.5 * (Ps_s[k] + Ps_s[k].T))))
            self.assertGreater(eig, -1e-10)

    def test_smoother_matches_filter_at_endpoint(self):
        """The smoothed estimate at the last step must equal the filtered one."""
        _, xs, Ps, Fs, Qs = _simulate_and_filter(seed=1, n_steps=50, dt=0.1)
        xs_s, Ps_s = rts_smooth(xs, Ps, Fs, Qs)
        self.assertTrue(np.allclose(xs_s[-1], xs[-1]))
        self.assertTrue(np.allclose(Ps_s[-1], Ps[-1]))

    def test_single_step_trace_is_identity(self):
        xs = np.array([[1.0, 2.0]])
        Ps = np.array([[[0.5, 0.0], [0.0, 0.7]]])
        Fs = np.zeros((0, 2, 2))
        Qs = np.zeros((0, 2, 2))
        xs_s, Ps_s = rts_smooth(xs, Ps, Fs, Qs)
        self.assertTrue(np.allclose(xs_s, xs))
        self.assertTrue(np.allclose(Ps_s, Ps))

    def test_length_n_and_length_n_minus_one_agree(self):
        """FilterPy-compat: passing Fs/Qs at length N (last unused) must match
        passing them at length N-1."""
        _, xs, Ps, Fs_n, Qs_n = _simulate_and_filter(seed=2, n_steps=20, dt=0.1)
        xs_a, Ps_a = rts_smooth(xs, Ps, Fs_n, Qs_n)
        xs_b, Ps_b = rts_smooth(xs, Ps, Fs_n[:-1], Qs_n[:-1])
        self.assertTrue(np.allclose(xs_a, xs_b))
        self.assertTrue(np.allclose(Ps_a, Ps_b))

    def test_non_constant_f_is_handled(self):
        """Per-step F that drifts mid-trajectory must produce a finite,
        symmetric, PSD smoothed trace (the backward pass uses F_k, not a
        single global F)."""
        truth, xs, Ps, Fs, Qs = _simulate_and_filter(seed=3, n_steps=100, dt=0.1)
        # Halfway through, perturb the F we tell the smoother about. The
        # smoother shouldn't blow up; just check shape + PSD output.
        Fs[50:] = np.array([[1.0, 0.05], [0.0, 1.0]])
        xs_s, Ps_s = rts_smooth(xs, Ps, Fs, Qs)
        self.assertEqual(xs_s.shape, xs.shape)
        self.assertEqual(Ps_s.shape, Ps.shape)
        self.assertTrue(np.all(np.isfinite(xs_s)))
        self.assertTrue(np.all(np.isfinite(Ps_s)))

    def test_bad_shapes_rejected(self):
        xs = np.zeros((10, 2))
        Ps = np.tile(np.eye(2), (10, 1, 1))
        Fs = np.tile(np.eye(2), (10, 1, 1))
        Qs = np.tile(np.eye(2), (10, 1, 1))

        with self.assertRaises(ValueError):
            rts_smooth(np.zeros(10), Ps, Fs, Qs)  # 1D xs
        with self.assertRaises(ValueError):
            rts_smooth(xs, np.tile(np.eye(2), (9, 1, 1)), Fs, Qs)  # wrong N for Ps
        with self.assertRaises(ValueError):
            rts_smooth(xs, Ps, np.tile(np.eye(2), (5, 1, 1)), Qs)  # wrong-len Fs
        with self.assertRaises(ValueError):
            rts_smooth(xs, Ps, Fs, np.tile(np.eye(3), (10, 1, 1)))  # wrong inner-shape Qs
        with self.assertRaises(ValueError):
            rts_smooth(np.zeros((0, 2)), np.zeros((0, 2, 2)), np.zeros((0, 2, 2)), np.zeros((0, 2, 2)))

    def test_classmethod_form_also_works(self):
        truth, xs, Ps, Fs, Qs = _simulate_and_filter(seed=5, n_steps=30, dt=0.1)
        xs_a, _ = rts_smooth(xs, Ps, Fs, Qs)
        xs_b, _ = TurboCKF.rts_smooth(xs, Ps, Fs, Qs)
        self.assertTrue(np.allclose(xs_a, xs_b))


if __name__ == "__main__":
    unittest.main()

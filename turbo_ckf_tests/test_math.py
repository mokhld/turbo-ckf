import unittest

import numpy as np

from turbo_ckf import TurboCKF

try:
    from filterpy.kalman import CubatureKalmanFilter as FilterPyCKF
except Exception:  # pragma: no cover - optional dependency
    FilterPyCKF = None


def linear_fx(x, dt):
    f = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    if x.ndim == 2:
        return x @ f.T
    return f @ x


def linear_hx(x):
    h = np.array([[1.0, 0.0]], dtype=float)
    if x.ndim == 2:
        return x @ h.T
    return h @ x


def kf_predict(x, p, f, q):
    xp = f @ x
    pp = f @ p @ f.T + q
    return xp, pp


def kf_update(x, p, z, h, r):
    y = z - h @ x
    s = h @ p @ h.T + r
    k = p @ h.T @ np.linalg.inv(s)
    xn = x + k @ y
    i = np.eye(x.shape[0], dtype=float)
    pn = (i - k @ h) @ p
    return xn, pn


class MathTests(unittest.TestCase):
    @unittest.skipIf(FilterPyCKF is None, "filterpy not installed")
    def test_linear_model_matches_filterpy_ckf(self):
        dt = 0.1
        q = np.array([[1e-3, 0.0], [0.0, 1e-3]], dtype=float)
        r = np.array([[5e-2]], dtype=float)

        ckf = TurboCKF(dim_x=2, dim_z=1, dt=dt, hx=linear_hx, fx=linear_fx)
        fp = FilterPyCKF(dim_x=2, dim_z=1, dt=dt, hx=lambda x: x[:1], fx=lambda x, dt: np.array([x[0] + dt * x[1], x[1]], dtype=float))

        ckf.x = np.array([0.0, 1.0], dtype=float)
        ckf.P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        ckf.Q = q.copy()
        ckf.R = r.copy()

        fp.x = ckf.x.copy()
        fp.P = ckf.P.copy()
        fp.Q = q.copy()
        fp.R = r.copy()

        for k in range(100):
            z = np.array([0.1 * (k + 1)], dtype=float)
            ckf.predict()
            ckf.update(z)
            fp.predict()
            fp.update(z)

            self.assertTrue(np.allclose(ckf.x, np.asarray(fp.x).reshape(-1), atol=1e-10))
            self.assertTrue(np.allclose(ckf.P, fp.P, atol=1.1e-3))

    def test_optimized_predict_covariance_matches_standard_form(self):
        dt = 0.2
        ckf = TurboCKF(dim_x=3, dim_z=1, dt=dt, hx=lambda x: x[..., :1], fx=lambda x, dt: x + dt)
        ckf.x = np.array([1.0, -2.0, 0.5], dtype=float)
        ckf.P = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, 0.0], [0.1, 0.0, 1.1]], dtype=float)
        ckf.Q = np.zeros((3, 3), dtype=float)

        sigma = ckf._cubature_points(ckf.x, ckf.P)
        transformed = ckf._apply_transition(ckf.fx, sigma, ckf.dt, ())
        m = transformed.shape[0]
        w = 1.0 / m
        mean = w * np.sum(transformed, axis=0)

        p_standard = np.zeros((3, 3), dtype=float)
        for i in range(m):
            d = transformed[i] - mean
            p_standard += w * np.outer(d, d)

        ckf.predict()
        self.assertTrue(np.allclose(ckf.x, mean, atol=1e-12))
        self.assertTrue(np.allclose(ckf.P, p_standard, atol=1e-12))

    def test_paper_kckf_prediction_matches_ckf_prediction(self):
        dt = 0.1
        kckf = TurboCKF(dim_x=4, dim_z=2, dt=dt, hx=lambda x: x[..., :2], fx=lambda x, dt: x)
        ckf = TurboCKF(dim_x=4, dim_z=2, dt=dt, hx=lambda x: x[..., :2], fx=lambda x, dt: x)

        x0 = np.array([0.0, 1.0, 10.0, -2.0], dtype=float)
        p0 = np.array(
            [
                [2.0, 0.3, 0.0, 0.0],
                [0.3, 1.2, 0.1, 0.0],
                [0.0, 0.1, 1.8, 0.2],
                [0.0, 0.0, 0.2, 1.1],
            ],
            dtype=float,
        )
        q = 1e-4 * np.eye(4, dtype=float)

        for f in (kckf, ckf):
            f.x = x0.copy()
            f.P = p0.copy()
            f.Q = q.copy()

        for _ in range(100):
            kckf.predict_standard_model("constant_velocity")
            ckf.predict_standard_model_ckf("constant_velocity")

        self.assertTrue(np.allclose(kckf.x, ckf.x, atol=1e-10))
        self.assertTrue(np.allclose(kckf.P, ckf.P, atol=1e-10))


if __name__ == "__main__":
    unittest.main()

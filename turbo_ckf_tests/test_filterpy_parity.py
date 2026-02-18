import unittest

import numpy as np

from turbo_ckf import TurboCKF

try:
    from filterpy.kalman import CubatureKalmanFilter as FilterPyCKF
except Exception:  # pragma: no cover - optional dependency
    FilterPyCKF = None


def fx(x, dt):
    if x.ndim == 2:
        out = x.copy()
        out[:, 0] = x[:, 0] + dt * x[:, 1]
        return out
    return np.array([x[0] + dt * x[1], x[1]], dtype=float)


def hx(x):
    if x.ndim == 2:
        return x[:, :1]
    return x[:1]


@unittest.skipIf(FilterPyCKF is None, "filterpy not installed")
class FilterPyParityTests(unittest.TestCase):
    def test_state_tracks_filterpy(self):
        dt = 0.1
        turbo = TurboCKF(dim_x=2, dim_z=1, dt=dt, hx=hx, fx=fx)
        fp = FilterPyCKF(dim_x=2, dim_z=1, dt=dt, hx=lambda x: x[:1], fx=lambda x, dt: np.array([x[0] + dt * x[1], x[1]]))

        x0 = np.array([0.0, 1.0], dtype=float)
        p0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        q = np.array([[1e-3, 0.0], [0.0, 1e-3]], dtype=float)
        r = np.array([[1e-2]], dtype=float)

        turbo.x = x0.copy()
        turbo.P = p0.copy()
        turbo.Q = q.copy()
        turbo.R = r.copy()

        fp.x = x0.copy()
        fp.P = p0.copy()
        fp.Q = q.copy()
        fp.R = r.copy()

        for k in range(100):
            z = np.array([0.05 * k], dtype=float)
            turbo.predict()
            turbo.update(z)
            fp.predict()
            fp.update(z)

        self.assertTrue(np.allclose(turbo.x.reshape(-1), np.asarray(fp.x, dtype=float).reshape(-1), atol=1e-8))


if __name__ == "__main__":
    unittest.main()

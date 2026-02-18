import unittest

import numpy as np

from turbo_ckf import TurboCKF


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


class RustBackendTests(unittest.TestCase):
    def test_rust_extension_is_available(self):
        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        self.assertTrue(hasattr(ckf, "_backend_name"))
        self.assertEqual(ckf._backend_name, "rust")

    def test_rust_backend_predict_update(self):
        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        ckf.x = np.array([0.0, 1.0], dtype=float)
        ckf.P = np.eye(2, dtype=float)
        ckf.Q = 1e-3 * np.eye(2, dtype=float)
        ckf.R = np.array([[1e-2]], dtype=float)

        for k in range(25):
            z = np.array([0.1 * k], dtype=float)
            ckf.predict()
            ckf.update(z)

        self.assertEqual(ckf.x.shape, (2,))
        self.assertEqual(ckf.P.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(ckf.x)))
        self.assertTrue(np.all(np.isfinite(ckf.P)))


if __name__ == "__main__":
    unittest.main()

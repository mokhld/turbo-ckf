import inspect
import unittest

import numpy as np

from turbo_ckf import TurboCKF


def fx(x, dt):
    if x.ndim == 2:
        out = x.copy()
        out[:, 0] = x[:, 0] + dt * x[:, 1]
        return out
    out = x.copy()
    out[0] = x[0] + dt * x[1]
    return out


def hx(x):
    if x.ndim == 2:
        return x[:, :1]
    return x[:1]


class APITests(unittest.TestCase):
    def test_constructor(self):
        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        self.assertEqual(ckf.dim_x, 2)
        self.assertEqual(ckf.dim_z, 1)
        self.assertAlmostEqual(ckf.dt, 0.1)
        self.assertEqual(ckf.x.shape, (2,))
        self.assertEqual(ckf.P.shape, (2, 2))
        self.assertEqual(ckf.Q.shape, (2, 2))
        self.assertEqual(ckf.R.shape, (1, 1))

    def test_predict_update_signatures(self):
        self.assertEqual(str(inspect.signature(TurboCKF.predict)), "(self, dt=None, fx=None, fx_args=())")
        self.assertEqual(str(inspect.signature(TurboCKF.update)), "(self, z, R=None, hx=None, hx_args=())")

    def test_core_state_attributes_exist(self):
        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        for attr in ["x", "P", "Q", "R", "K", "y", "z", "S", "SI", "x_prior", "P_prior", "x_post", "P_post"]:
            self.assertTrue(hasattr(ckf, attr), f"missing attribute {attr}")

    def test_predict_then_update_runs(self):
        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        ckf.x = np.array([0.0, 1.0], dtype=float)
        ckf.P = np.eye(2)
        ckf.Q = 0.01 * np.eye(2)
        ckf.R = np.array([[0.1]], dtype=float)

        ckf.predict()
        ckf.update(np.array([0.05], dtype=float))

        self.assertEqual(ckf.x.shape, (2,))
        self.assertEqual(ckf.P.shape, (2, 2))
        self.assertEqual(ckf.z.shape, (1,))

    def test_predict_and_update_pass_through_args(self):
        seen = {"fx": None, "hx": None}

        def fx_with_args(x, dt, gain):
            seen["fx"] = gain
            if x.ndim == 2:
                out = x.copy()
                out[:, 0] = x[:, 0] + gain * dt * x[:, 1]
                return out
            return np.array([x[0] + gain * dt * x[1], x[1]], dtype=float)

        def hx_with_args(x, offset):
            seen["hx"] = offset
            if x.ndim == 2:
                return x[:, :1] + offset
            return x[:1] + offset

        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_with_args, fx=fx_with_args)
        ckf.predict(fx_args=(2.0,))
        ckf.update(np.array([1.0], dtype=float), hx_args=(0.5,))
        self.assertEqual(seen["fx"], 2.0)
        self.assertEqual(seen["hx"], 0.5)

    def test_update_accepts_scalar_R(self):
        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        ckf.predict()
        ckf.update(np.array([0.0], dtype=float), R=0.25)
        self.assertTrue(np.allclose(ckf.S, ckf.S.T))

    def test_update_none_keeps_state(self):
        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        ckf.x = np.array([2.0, 3.0], dtype=float)
        ckf.P = np.eye(2)
        before_x = ckf.x.copy()
        before_p = ckf.P.copy()
        ckf.update(None)
        self.assertTrue(np.allclose(ckf.x, before_x))
        self.assertTrue(np.allclose(ckf.P, before_p))


if __name__ == "__main__":
    unittest.main()

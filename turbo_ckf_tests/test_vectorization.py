import unittest

import numpy as np

from turbo_ckf import TurboCKF


def hx(x):
    if x.ndim == 2:
        return x[:, :1]
    return x[:1]


class VectorizationTests(unittest.TestCase):
    def test_predict_uses_single_vectorized_fx_call_when_available(self):
        calls = {"count": 0}

        def fx(x, dt):
            calls["count"] += 1
            if x.ndim != 2:
                raise AssertionError("expected batched input")
            out = x.copy()
            out[:, 0] = x[:, 0] + dt * x[:, 1]
            return out

        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        ckf.predict()
        self.assertEqual(calls["count"], 1)

    def test_predict_requires_vectorized_fx(self):
        calls = {"count": 0}

        def fx(x, dt):
            calls["count"] += 1
            if x.ndim == 2:
                raise TypeError("pointwise only")
            return np.array([x[0] + dt * x[1], x[1]], dtype=float)

        ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)
        with self.assertRaises(TypeError):
            ckf.predict()
        self.assertEqual(calls["count"], 1)

    def test_predict_standard_model_constant_velocity(self):
        ckf = TurboCKF(dim_x=4, dim_z=2, dt=0.5, hx=lambda x: x[..., :2], fx=lambda x, dt: x)
        ckf.x = np.array([0.0, 10.0, 1.0, -2.0], dtype=float)
        ckf.predict_standard_model(model_type="constant_velocity")
        self.assertTrue(np.allclose(ckf.x, np.array([0.5, 9.0, 1.0, -2.0], dtype=float), atol=1e-12))

    def test_predict_standard_model_constant_acceleration(self):
        ckf = TurboCKF(dim_x=6, dim_z=2, dt=2.0, hx=lambda x: x[..., :2], fx=lambda x, dt: x)
        ckf.x = np.array([0.0, 0.0, 1.0, -1.0, 0.5, 2.0], dtype=float)
        ckf.predict_standard_model(model_type="constant_acceleration")
        expected = np.array([3.0, 2.0, 2.0, 3.0, 0.5, 2.0], dtype=float)
        self.assertTrue(np.allclose(ckf.x, expected, atol=1e-12))

    def test_predict_standard_model_raises_for_bad_layout(self):
        ckf = TurboCKF(dim_x=5, dim_z=1, dt=1.0, hx=hx, fx=lambda x, dt: x)
        with self.assertRaises(ValueError):
            ckf.predict_standard_model(model_type="constant_velocity")

    def test_predict_standard_model_ckf_matches_kckf(self):
        kckf = TurboCKF(dim_x=4, dim_z=2, dt=0.2, hx=lambda x: x[..., :2], fx=lambda x, dt: x)
        ckf = TurboCKF(dim_x=4, dim_z=2, dt=0.2, hx=lambda x: x[..., :2], fx=lambda x, dt: x)

        x0 = np.array([1.0, -2.0, 0.7, 3.0], dtype=float)
        p0 = np.array(
            [
                [1.4, 0.2, 0.0, 0.1],
                [0.2, 2.1, 0.2, 0.0],
                [0.0, 0.2, 1.3, 0.1],
                [0.1, 0.0, 0.1, 0.9],
            ],
            dtype=float,
        )
        q = 1e-3 * np.eye(4, dtype=float)

        for f in (kckf, ckf):
            f.x = x0.copy()
            f.P = p0.copy()
            f.Q = q.copy()

        kckf.predict_standard_model("constant_velocity")
        ckf.predict_standard_model_ckf("constant_velocity")

        self.assertTrue(np.allclose(kckf.x, ckf.x, atol=1e-12))
        self.assertTrue(np.allclose(kckf.P, ckf.P, atol=1e-12))

    def test_predict_linear_model_matches_standard_model(self):
        dt = 0.1
        f = np.eye(4, dtype=float)
        f[0, 2] = dt
        f[1, 3] = dt

        a = TurboCKF(dim_x=4, dim_z=2, dt=dt, hx=lambda x: x[..., :2], fx=lambda x, dt: x)
        b = TurboCKF(dim_x=4, dim_z=2, dt=dt, hx=lambda x: x[..., :2], fx=lambda x, dt: x)

        x0 = np.array([0.0, 10.0, 1.0, -2.0], dtype=float)
        p0 = np.array(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.1, 1.3, 0.2, 0.1],
                [0.0, 0.2, 0.9, 0.0],
                [0.0, 0.1, 0.0, 1.2],
            ],
            dtype=float,
        )
        q = 1e-3 * np.eye(4, dtype=float)

        for ckf in (a, b):
            ckf.x = x0.copy()
            ckf.P = p0.copy()
            ckf.Q = q.copy()

        a.predict_standard_model("constant_velocity")
        b.predict_linear_model(f)

        self.assertTrue(np.allclose(a.x, b.x, atol=1e-12))
        self.assertTrue(np.allclose(a.P, b.P, atol=1e-12))

    def test_predict_linear_model_ckf_matches_standard_model_ckf(self):
        dt = 0.1
        f = np.eye(4, dtype=float)
        f[0, 2] = dt
        f[1, 3] = dt

        a = TurboCKF(dim_x=4, dim_z=2, dt=dt, hx=lambda x: x[..., :2], fx=lambda x, dt: x)
        b = TurboCKF(dim_x=4, dim_z=2, dt=dt, hx=lambda x: x[..., :2], fx=lambda x, dt: x)

        x0 = np.array([1.0, -2.0, 0.5, 3.0], dtype=float)
        p0 = np.array(
            [
                [1.2, 0.1, 0.0, 0.0],
                [0.1, 1.0, 0.2, 0.0],
                [0.0, 0.2, 1.4, 0.1],
                [0.0, 0.0, 0.1, 1.1],
            ],
            dtype=float,
        )
        q = 1e-3 * np.eye(4, dtype=float)

        for ckf in (a, b):
            ckf.x = x0.copy()
            ckf.P = p0.copy()
            ckf.Q = q.copy()

        a.predict_standard_model_ckf("constant_velocity")
        b.predict_linear_model_ckf(f)

        self.assertTrue(np.allclose(a.x, b.x, atol=1e-12))
        self.assertTrue(np.allclose(a.P, b.P, atol=1e-12))

    def test_normalize_state_quaternion(self):
        ckf = TurboCKF(dim_x=4, dim_z=2, dt=0.1, hx=lambda x: x[..., :2], fx=lambda x, dt: x)
        ckf.x = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)
        ckf.normalize_state_quaternion()
        self.assertTrue(np.allclose(ckf.x, np.array([1.0, 0.0, 0.0, 0.0], dtype=float), atol=1e-12))


if __name__ == "__main__":
    unittest.main()

"""State layout handling in predict_standard_model[_ckf].

"blocked" is [pos..., vel...(, acc...)] and "interleaved" is FilterPy's
per-axis [x, vx, (ax,) y, vy, (ay,) ...]. The reference F matrices are built
with np.kron so they do not share code with the Rust transition_matrix.
"""

import unittest

import numpy as np

from turbo_ckf import TurboCKF


def axis_block(model_type, dt):
    if model_type == "constant_velocity":
        return np.array([[1.0, dt], [0.0, 1.0]])
    return np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])


def reference_f(model_type, layout, n_axes, dt):
    block = axis_block(model_type, dt)
    eye = np.eye(n_axes)
    if layout == "interleaved":
        return np.kron(eye, block)
    return np.kron(block, eye)


def make_filter(dim_x, dt, seed=0):
    kf = TurboCKF(dim_x=dim_x, dim_z=1, dt=dt, hx=lambda x: x[..., :1], fx=lambda x, dt: x)
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim_x, dim_x))
    kf.x = rng.normal(size=dim_x)
    kf.P = a @ a.T + dim_x * np.eye(dim_x)
    kf.Q = 1e-3 * np.eye(dim_x)
    return kf


CASES = [
    ("constant_velocity", 2),
    ("constant_velocity", 3),
    ("constant_acceleration", 2),
    ("constant_acceleration", 3),
]

# (standard-model method, matching explicit-F method)
VARIANTS = [
    ("predict_standard_model", "predict_linear_model"),
    ("predict_standard_model_ckf", "predict_linear_model_ckf"),
]


class InterleavedLayoutTests(unittest.TestCase):
    def test_interleaved_cv_does_not_mix_axes(self):
        # Regression for REVIEW A6: the blocked-only F moved x by y (x = 100).
        for method, _ in VARIANTS:
            with self.subTest(method=method):
                kf = TurboCKF(dim_x=4, dim_z=2, dt=1.0, hx=lambda x: x[..., ::2], fx=lambda x, dt: x)
                kf.x = [0.0, 1.0, 100.0, 0.0]  # [x, vx, y, vy]
                getattr(kf, method)("constant_velocity", layout="interleaved")
                np.testing.assert_allclose(kf.x, [1.0, 1.0, 100.0, 0.0], atol=1e-9)

    def test_interleaved_ca_does_not_mix_axes(self):
        for method, _ in VARIANTS:
            with self.subTest(method=method):
                kf = TurboCKF(dim_x=6, dim_z=2, dt=2.0, hx=lambda x: x[..., ::3], fx=lambda x, dt: x)
                kf.x = [0.0, 1.0, 0.5, 100.0, 0.0, 0.0]  # [x, vx, ax, y, vy, ay]
                getattr(kf, method)("constant_acceleration", layout="interleaved")
                # x = 0 + 1*2 + 0.5*0.5*4 = 3, vx = 1 + 0.5*2 = 2
                np.testing.assert_allclose(kf.x, [3.0, 2.0, 0.5, 100.0, 0.0, 0.0], atol=1e-9)

    def test_interleaved_matches_explicit_f(self):
        dt = 0.1
        for model_type, n_axes in CASES:
            order = 2 if model_type == "constant_velocity" else 3
            dim_x = order * n_axes
            f = reference_f(model_type, "interleaved", n_axes, dt)
            for method, linear_method in VARIANTS:
                with self.subTest(model=model_type, axes=n_axes, method=method):
                    a = make_filter(dim_x, dt)
                    b = make_filter(dim_x, dt)
                    getattr(a, method)(model_type, layout="interleaved")
                    getattr(b, linear_method)(f)
                    np.testing.assert_allclose(a.x, b.x, atol=1e-12)
                    np.testing.assert_allclose(a.P, b.P, atol=1e-12)


class BlockedLayoutTests(unittest.TestCase):
    def test_default_layout_is_blocked_and_matches_explicit_f(self):
        dt = 0.1
        for model_type, n_axes in CASES:
            order = 2 if model_type == "constant_velocity" else 3
            dim_x = order * n_axes
            f = reference_f(model_type, "blocked", n_axes, dt)
            for method, linear_method in VARIANTS:
                with self.subTest(model=model_type, axes=n_axes, method=method):
                    default = make_filter(dim_x, dt)
                    blocked = make_filter(dim_x, dt)
                    explicit = make_filter(dim_x, dt)
                    getattr(default, method)(model_type)
                    getattr(blocked, method)(model_type, layout="blocked")
                    getattr(explicit, linear_method)(f)
                    np.testing.assert_array_equal(default.x, blocked.x)
                    np.testing.assert_array_equal(default.P, blocked.P)
                    np.testing.assert_allclose(default.x, explicit.x, atol=1e-12)
                    np.testing.assert_allclose(default.P, explicit.P, atol=1e-12)


class LayoutValidationTests(unittest.TestCase):
    def test_unknown_layout_raises_and_lists_choices(self):
        for method, _ in VARIANTS:
            with self.subTest(method=method):
                kf = make_filter(4, 0.1)
                x0, p0 = kf.x.copy(), kf.P.copy()
                with self.assertRaisesRegex(ValueError, r"'blocked', 'interleaved'"):
                    getattr(kf, method)("constant_velocity", layout="filterpy")
                np.testing.assert_array_equal(kf.x, x0)
                np.testing.assert_array_equal(kf.P, p0)

    def test_backend_rejects_unknown_layout(self):
        kf = make_filter(4, 0.1)
        for name in ("predict_standard_model", "predict_standard_model_ckf"):
            with self.subTest(method=name):
                with self.assertRaisesRegex(ValueError, "unsupported layout"):
                    getattr(kf._rust_backend, name)("constant_velocity", "filterpy")

    def test_interleaved_dimension_mismatch_raises(self):
        for model_type, dim_x in (("constant_velocity", 5), ("constant_acceleration", 4)):
            with self.subTest(model=model_type):
                kf = make_filter(dim_x, 0.1)
                with self.assertRaisesRegex(ValueError, "interleaved"):
                    kf.predict_standard_model(model_type, layout="interleaved")


if __name__ == "__main__":
    unittest.main()

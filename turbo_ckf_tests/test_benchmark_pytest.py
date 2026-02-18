from __future__ import annotations

import numpy as np
from filterpy.kalman import CubatureKalmanFilter as FilterPyCKF

from turbo_ckf import TurboCKF


def fx_vectorized(x, dt):
    if x.ndim == 2:
        out = x.copy()
        out[:, 0] = x[:, 0] + dt * x[:, 1]
        return out
    return np.array([x[0] + dt * x[1], x[1]], dtype=float)


def fx_pointwise(x, dt):
    return np.array([x[0] + dt * x[1], x[1]], dtype=float)


def hx_vectorized(x):
    if x.ndim == 2:
        return x[:, :1]
    return x[:1]


def hx_pointwise(x):
    return x[:1]


def _run_filter(kf, steps=2_000):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        kf.predict()
        kf.update(z)


def _run_turbo_model(kf, steps=2_000):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        kf.predict_standard_model("constant_velocity")
        kf.update(z)


def _run_turbo_model_ckf(kf, steps=2_000):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        kf.predict_standard_model_ckf("constant_velocity")
        kf.update(z)


def test_benchmark_filterpy(benchmark):
    kf = FilterPyCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pointwise, fx=fx_pointwise)
    kf.x = np.array([0.0, 1.0], dtype=float)
    kf.P = np.eye(2) * 0.5
    kf.Q = np.eye(2) * 1e-3
    kf.R = np.eye(1) * 1e-2
    benchmark(_run_filter, kf)


def test_benchmark_turbo_callback(benchmark):
    kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_vectorized, fx=fx_vectorized)
    kf.x = np.array([0.0, 1.0], dtype=float)
    kf.P = np.eye(2) * 0.5
    kf.Q = np.eye(2) * 1e-3
    kf.R = np.eye(1) * 1e-2
    benchmark(_run_filter, kf)


def test_benchmark_turbo_standard_model(benchmark):
    kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_vectorized, fx=fx_vectorized)
    kf.x = np.array([0.0, 1.0], dtype=float)
    kf.P = np.eye(2) * 0.5
    kf.Q = np.eye(2) * 1e-3
    kf.R = np.eye(1) * 1e-2
    benchmark(_run_turbo_model, kf)


def test_benchmark_turbo_standard_model_ckf(benchmark):
    kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_vectorized, fx=fx_vectorized)
    kf.x = np.array([0.0, 1.0], dtype=float)
    kf.P = np.eye(2) * 0.5
    kf.Q = np.eye(2) * 1e-3
    kf.R = np.eye(1) * 1e-2
    benchmark(_run_turbo_model_ckf, kf)

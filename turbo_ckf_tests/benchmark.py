"""Scenario benchmark script matching the Turbo-CKF spec."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from filterpy.kalman import CubatureKalmanFilter as FilterPyCKF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def make_filterpy():
    kf = FilterPyCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pointwise, fx=fx_pointwise)
    kf.x = np.array([0.0, 1.0], dtype=float)
    kf.P = np.eye(2) * 0.5
    kf.Q = np.eye(2) * 1e-3
    kf.R = np.eye(1) * 1e-2
    return kf


def make_turbo():
    ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_vectorized, fx=fx_vectorized)
    ckf.x = np.array([0.0, 1.0], dtype=float)
    ckf.P = np.eye(2) * 0.5
    ckf.Q = np.eye(2) * 1e-3
    ckf.R = np.eye(1) * 1e-2
    return ckf


def run_filter(kf, steps):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        kf.predict()
        kf.update(z)


def run_turbo_model(ckf, steps):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        ckf.predict_standard_model("constant_velocity")
        ckf.update(z)


def run_turbo_model_ckf(ckf, steps):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        ckf.predict_standard_model_ckf("constant_velocity")
        ckf.update(z)


def run_predict_only_backend(backend, mode: str, steps: int):
    fn = backend.predict_standard_model if mode == "kckf" else backend.predict_standard_model_ckf
    for _ in range(steps):
        fn("constant_velocity")


def timed(label, fn, steps):
    t0 = time.perf_counter()
    fn(steps)
    dt = time.perf_counter() - t0
    print(f"{label:40s}{dt:.4f}s")
    return dt


def main():
    steps = 50_000
    fp = make_filterpy()
    turbo_cb = make_turbo()
    turbo_model = make_turbo()
    turbo_model_ckf = make_turbo()

    t_fp = timed("FilterPy (pure Python)", lambda n: run_filter(fp, n), steps)
    t_cb = timed("TurboCKF (Rust + Python callback)", lambda n: run_filter(turbo_cb, n), steps)
    t_model = timed("TurboCKF (pure Rust motion model)", lambda n: run_turbo_model(turbo_model, n), steps)
    t_model_ckf = timed("TurboCKF (standard CKF prediction)", lambda n: run_turbo_model_ckf(turbo_model_ckf, n), steps)

    print("")
    print(f"Scenario2 speedup vs FilterPy: {t_fp / t_cb:.2f}x")
    print(f"Scenario3 speedup vs FilterPy: {t_fp / t_model:.2f}x")
    print(f"KCKF-vs-CKF speedup (full step): {t_model_ckf / t_model:.2f}x")

    print("")
    print("Equation-aligned prediction-only benchmark")
    paper_steps = 400_000
    kckf = make_turbo()
    ckf = make_turbo()
    backend_kckf = kckf._rust_backend
    backend_ckf = ckf._rust_backend
    backend_kckf.set_state(kckf.x, kckf.P, kckf.Q, kckf.R)
    backend_ckf.set_state(ckf.x, ckf.P, ckf.Q, ckf.R)
    t_pred_kckf = timed("Prediction only (KCKF Eq.16-19)", lambda n: run_predict_only_backend(backend_kckf, "kckf", n), paper_steps)
    t_pred_ckf = timed("Prediction only (CKF Eq.20-23)", lambda n: run_predict_only_backend(backend_ckf, "ckf", n), paper_steps)
    print(f"KCKF-vs-CKF speedup (prediction only): {t_pred_ckf / t_pred_kckf:.2f}x")


if __name__ == "__main__":
    main()

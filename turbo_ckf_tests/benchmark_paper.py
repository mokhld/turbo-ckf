"""Benchmark aligned with the KCKF AHRS equation set."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_ckf import TurboCKF
from turbo_ckf.paper_ahrs import (
    measurement_noise,
    normalize_quaternion,
    observation_model,
    transition_matrix_from_gyro,
)


DT = 0.01
SIGMA_ACC2 = 1e-2
SIGMA_MAG2 = 1e-2
MN = 0.8
MD = 0.6


def omega_at(step: int) -> np.ndarray:
    return np.array(
        [
            0.6 * np.sin(0.004 * step),
            0.3 * np.cos(0.006 * step),
            0.4 * np.sin(0.005 * step + 0.3),
        ],
        dtype=float,
    )


def make_filter() -> TurboCKF:
    ckf = TurboCKF(dim_x=4, dim_z=6, dt=DT, hx=observation_model, fx=lambda x, dt: x)
    ckf.x = normalize_quaternion(np.array([1.0, 0.03, -0.02, 0.01], dtype=float))
    ckf.P = 1e-2 * np.eye(4, dtype=float)
    ckf.Q = 1e-3 * np.eye(4, dtype=float)
    ckf.R = measurement_noise(SIGMA_ACC2, SIGMA_MAG2)
    return ckf


def build_transitions(steps: int) -> list[np.ndarray]:
    return [transition_matrix_from_gyro(omega_at(i), DT) for i in range(steps)]


def run_prediction_only_backend(backend, mode: str, transitions):
    fn = backend.predict_linear_model if mode == "kckf" else backend.predict_linear_model_ckf
    for f in transitions:
        fn(f)


def run_full_step_backend(backend, mode: str, transitions, z: np.ndarray):
    fn = backend.predict_linear_model if mode == "kckf" else backend.predict_linear_model_ckf
    for f in transitions:
        fn(f)
        backend.update_paper_ahrs(z, SIGMA_ACC2, SIGMA_MAG2)
        backend.normalize_quaternion_state()


def init_backend(ckf: TurboCKF):
    backend = ckf._rust_backend
    backend.set_state(ckf.x, ckf.P, ckf.Q, ckf.R)
    return backend


def run_full_step_wrapper(ckf: TurboCKF, mode: str, transitions, z: np.ndarray):
    for f in transitions:
        if mode == "kckf":
            ckf.predict_linear_model(f)
        else:
            ckf.predict_linear_model_ckf(f)
        ckf.update_paper_ahrs(z, SIGMA_ACC2, SIGMA_MAG2)
        ckf.normalize_state_quaternion_backend()


def timed(label, fn):
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    print(f"{label:45s}{dt:.4f}s")
    return dt


def main():
    z_ref = observation_model(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), MN, MD)
    r = measurement_noise(SIGMA_ACC2, SIGMA_MAG2)

    print("Equation-aligned prediction benchmark (Eq.16-23)")
    pred_steps = 250_000
    transitions_pred = build_transitions(pred_steps)
    kckf_pred = make_filter()
    ckf_pred = make_filter()
    backend_k = init_backend(kckf_pred)
    backend_c = init_backend(ckf_pred)
    t_pred_k = timed(
        "Prediction only (KCKF Eq.16-19, backend)",
        lambda: run_prediction_only_backend(backend_k, "kckf", transitions_pred),
    )
    t_pred_c = timed(
        "Prediction only (CKF Eq.20-23, backend)",
        lambda: run_prediction_only_backend(backend_c, "ckf", transitions_pred),
    )
    print(f"KCKF-vs-CKF speedup (prediction only): {t_pred_c / t_pred_k:.2f}x")

    print("")
    print("Equation-aligned full-step benchmark (prediction + update)")
    full_steps = 30_000
    transitions_full = build_transitions(full_steps)
    kckf_full = make_filter()
    ckf_full = make_filter()
    backend_full_k = init_backend(kckf_full)
    backend_full_c = init_backend(ckf_full)
    t_full_k = timed(
        "Full step (KCKF prediction, backend)",
        lambda: run_full_step_backend(backend_full_k, "kckf", transitions_full, z_ref),
    )
    t_full_c = timed(
        "Full step (CKF prediction, backend)",
        lambda: run_full_step_backend(backend_full_c, "ckf", transitions_full, z_ref),
    )
    print(f"KCKF-vs-CKF speedup (full step, backend): {t_full_c / t_full_k:.2f}x")

    print("")
    print("Wrapper-path benchmark (includes Python state sync + normalization)")
    wrap_steps = 20_000
    transitions_wrap = transitions_full[:wrap_steps]
    wrap_k = make_filter()
    wrap_c = make_filter()
    t_wrap_k = timed(
        "Full step (KCKF prediction, wrapper)",
        lambda: run_full_step_wrapper(wrap_k, "kckf", transitions_wrap, z_ref),
    )
    t_wrap_c = timed(
        "Full step (CKF prediction, wrapper)",
        lambda: run_full_step_wrapper(wrap_c, "ckf", transitions_wrap, z_ref),
    )
    print(f"KCKF-vs-CKF speedup (full step, wrapper): {t_wrap_c / t_wrap_k:.2f}x")


if __name__ == "__main__":
    main()

"""Constant-velocity tracking quickstart for TurboCKF.

Run from the repo root:

    .venv-turbo-ckf/bin/python examples/quickstart_cv.py

The script tracks a 1-D point moving at constant velocity with noisy position
measurements, and prints the final estimate alongside the ground-truth.

TurboCKF requires *vectorized* callbacks. Both ``fx`` and ``hx`` receive a
sigma-point matrix of shape ``(2 * dim_x, dim_x)`` (one sigma point per row)
and must return arrays of shape ``(2 * dim_x, dim_x)`` and
``(2 * dim_x, dim_z)`` respectively.
"""

from __future__ import annotations

import numpy as np

from turbo_ckf import TurboCKF


def fx_vectorized(sigma_points: np.ndarray, dt: float) -> np.ndarray:
    """Constant-velocity transition applied to every sigma point at once.

    State layout: [position, velocity].

    Parameters
    ----------
    sigma_points:
        Array of shape ``(2 * dim_x, dim_x)``.
    dt:
        Time step in seconds.
    """
    out = np.empty_like(sigma_points)
    out[:, 0] = sigma_points[:, 0] + dt * sigma_points[:, 1]
    out[:, 1] = sigma_points[:, 1]
    return out


def hx_vectorized(sigma_points: np.ndarray) -> np.ndarray:
    """Measurement model: observe position only.

    Returns
    -------
    np.ndarray
        Array of shape ``(2 * dim_x, dim_z)`` with ``dim_z = 1``.
    """
    return sigma_points[:, 0:1]


def main() -> None:
    rng = np.random.default_rng(42)

    dim_x = 2
    dim_z = 1
    dt = 0.1
    n_steps = 200

    kf = TurboCKF(
        dim_x=dim_x,
        dim_z=dim_z,
        dt=dt,
        hx=hx_vectorized,
        fx=fx_vectorized,
    )
    kf.x = np.array([0.0, 0.0])
    kf.P = np.eye(dim_x) * 1.0
    kf.Q = np.eye(dim_x) * 1e-3
    kf.R = np.array([[0.25]])

    true_pos = 0.0
    true_vel = 1.5
    measurement_noise_std = 0.5

    last_estimate = kf.x.copy()
    for _ in range(n_steps):
        true_pos += true_vel * dt
        z = np.array([true_pos + rng.normal(0.0, measurement_noise_std)])

        kf.predict()
        kf.update(z)
        last_estimate = kf.x.copy()

    print(f"steps          : {n_steps}")
    print(f"true position  : {true_pos:.4f}")
    print(f"est. position  : {last_estimate[0]:.4f}")
    print(f"true velocity  : {true_vel:.4f}")
    print(f"est. velocity  : {last_estimate[1]:.4f}")


if __name__ == "__main__":
    main()

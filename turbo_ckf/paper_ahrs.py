"""AHRS model helpers implementing the KCKF equation set."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]


def normalize_quaternion(q: npt.ArrayLike) -> Vector:
    vec = np.asarray(q, dtype=float).reshape(-1)
    if vec.shape[0] != 4:
        raise ValueError(f"q must have length 4, got shape {vec.shape}")
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion norm must be finite and positive")
    return vec / norm


def transition_matrix_from_gyro(omega_xyz: npt.ArrayLike, dt: float) -> Matrix:
    """Equation (3): F_k = I + dt/2 * [Omega_x]."""

    omega = np.asarray(omega_xyz, dtype=float).reshape(-1)
    if omega.shape[0] != 3:
        raise ValueError(f"omega_xyz must have length 3, got shape {omega.shape}")
    wx, wy, wz = omega
    omega_cross = np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ],
        dtype=float,
    )
    return np.eye(4, dtype=float) + 0.5 * float(dt) * omega_cross


def process_noise_from_quaternion(q: npt.ArrayLike, dt: float, gyro_variance: float) -> Matrix:
    """Equations (4)-(5): Q_k = G_k * (sigma_w^2 I) * G_k^T."""

    q0, q1, q2, q3 = normalize_quaternion(q)
    g = 0.5 * float(dt) * np.array(
        [
            [q1, q2, q3],
            [q0, q3, -q2],
            [-q3, -q0, q1],
            [q2, -q1, -q0],
        ],
        dtype=float,
    )
    return float(gyro_variance) * (g @ g.T)


def measurement_noise(sigma_acc2: float, sigma_mag2: float) -> Matrix:
    """Equation (9)."""

    r = np.zeros((6, 6), dtype=float)
    r[:3, :3] = float(sigma_acc2) * np.eye(3, dtype=float)
    r[3:, 3:] = float(sigma_mag2) * np.eye(3, dtype=float)
    return r


def magnetic_reference_terms(accel_xyz: npt.ArrayLike, mag_xyz: npt.ArrayLike) -> tuple[float, float]:
    """Equations (12)-(13): m_D and m_N."""

    acc = np.asarray(accel_xyz, dtype=float).reshape(-1)
    mag = np.asarray(mag_xyz, dtype=float).reshape(-1)
    if acc.shape[0] != 3 or mag.shape[0] != 3:
        raise ValueError("accel_xyz and mag_xyz must have length 3")
    acc_norm = np.linalg.norm(acc)
    mag_norm = np.linalg.norm(mag)
    if acc_norm <= 0.0 or mag_norm <= 0.0:
        raise ValueError("accel_xyz and mag_xyz norms must be positive")
    acc_n = acc / acc_norm
    mag_n = mag / mag_norm
    m_d = float(np.clip(np.dot(acc_n, mag_n), -1.0, 1.0))
    m_n = float(np.sqrt(max(0.0, 1.0 - m_d * m_d)))
    return m_n, m_d


def observation_model(q: npt.ArrayLike, m_n: float, m_d: float):
    """Equation (14). Supports both vectorized and single-point input."""

    arr = np.asarray(q, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
        squeeze = True
    elif arr.ndim == 2 and arr.shape[1] == 4:
        squeeze = False
    else:
        raise ValueError(f"q must have shape (4,) or (N,4), got {arr.shape}")

    q0 = arr[:, 0]
    q1 = arr[:, 1]
    q2 = arr[:, 2]
    q3 = arr[:, 3]
    mn = float(m_n)
    md = float(m_d)

    out = np.empty((arr.shape[0], 6), dtype=float)
    out[:, 0] = 2.0 * (q1 * q3 - q0 * q2)
    out[:, 1] = 2.0 * (q2 * q3 + q0 * q1)
    out[:, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
    out[:, 3] = (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * mn + 2.0 * (q1 * q3 - q0 * q2) * md
    out[:, 4] = 2.0 * (q1 * q2 - q0 * q3) * mn + 2.0 * (q2 * q3 + q0 * q1) * md
    out[:, 5] = 2.0 * (q1 * q3 + q0 * q2) * mn + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * md

    if squeeze:
        return out[0]
    return out

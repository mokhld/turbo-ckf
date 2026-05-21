"""Turbo CKF package."""

from .core import TurboCKF
from .paper_ahrs import (
    magnetic_reference_terms,
    measurement_noise,
    normalize_quaternion,
    observation_model,
    process_noise_from_quaternion,
    transition_matrix_from_gyro,
)

rts_smooth = TurboCKF.rts_smooth

__all__ = [
    "TurboCKF",
    "rts_smooth",
    "magnetic_reference_terms",
    "measurement_noise",
    "normalize_quaternion",
    "observation_model",
    "process_noise_from_quaternion",
    "transition_matrix_from_gyro",
]

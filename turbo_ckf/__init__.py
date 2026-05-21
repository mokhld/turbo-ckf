"""Turbo CKF package."""

from .core import TurboCKF, TurboSRCKF
from .paper_ahrs import (
    magnetic_reference_terms,
    measurement_noise,
    normalize_quaternion,
    observation_model,
    process_noise_from_quaternion,
    transition_matrix_from_gyro,
)

rts_smooth = TurboCKF.rts_smooth
batch_filter = TurboCKF.batch_filter
batch_parallel_step = TurboCKF.batch_parallel_step

__all__ = [
    "TurboCKF",
    "TurboSRCKF",
    "batch_filter",
    "batch_parallel_step",
    "rts_smooth",
    "magnetic_reference_terms",
    "measurement_noise",
    "normalize_quaternion",
    "observation_model",
    "process_noise_from_quaternion",
    "transition_matrix_from_gyro",
]

"""update_paper_ahrs accepts accelerometer and magnetometer readings in any units.

The paper's observation model predicts unit vectors, so the Rust update scales
z[0:3] and z[3:6] to unit length before forming the innovation. These tests pin
that raw sensor units (m/s^2, uT) give the same posterior as pre-normalized
input, that a static attitude converges from raw readings, and that zero or
overflowing norms are rejected.
"""

import unittest

import numpy as np

from turbo_ckf import TurboCKF, normalize_quaternion, observation_model
from turbo_ckf.paper_ahrs import transition_matrix_from_gyro


def _ahrs_filter() -> TurboCKF:
    kf = TurboCKF(dim_x=4, dim_z=6, dt=0.01, hx=observation_model, fx=lambda x, dt: x)
    kf.x = normalize_quaternion(np.array([1.0, 0.04, -0.02, 0.03]))
    kf.P = 1e-2 * np.array(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.1, 1.2, 0.1, 0.0],
            [0.0, 0.1, 1.1, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ]
    )
    kf.Q = 1e-3 * np.eye(4)
    kf.predict_linear_model(transition_matrix_from_gyro(np.array([0.1, -0.2, 0.05]), 0.01))
    return kf


def _attitude_error_deg(q: np.ndarray, q_true: np.ndarray) -> float:
    return float(2.0 * np.degrees(np.arccos(min(1.0, abs(float(q @ q_true))))))


class PaperAhrsMeasurementUnitsTests(unittest.TestCase):
    def test_raw_units_match_unit_normalized_posterior(self):
        acc = np.array([0.31, -0.22, 9.74])  # m/s^2
        mag = np.array([22.0, -3.5, 41.0])  # uT
        z_raw = np.r_[acc, mag]
        z_unit = np.r_[acc / np.linalg.norm(acc), mag / np.linalg.norm(mag)]

        kf_raw = _ahrs_filter()
        kf_unit = _ahrs_filter()
        kf_raw.update_paper_ahrs(z_raw, sigma_acc2=1e-2, sigma_mag2=2e-2)
        kf_unit.update_paper_ahrs(z_unit, sigma_acc2=1e-2, sigma_mag2=2e-2)

        np.testing.assert_allclose(kf_raw.x, kf_unit.x, rtol=0, atol=1e-12)
        np.testing.assert_allclose(kf_raw.P, kf_unit.P, rtol=0, atol=1e-12)
        # The recorded measurement and innovation are the normalized ones.
        np.testing.assert_allclose(kf_raw.z, z_unit, rtol=0, atol=1e-15)
        np.testing.assert_allclose(kf_raw.y, kf_unit.y, rtol=0, atol=1e-12)
        np.testing.assert_allclose(kf_raw.y, z_unit - kf_raw.z_pred, rtol=0, atol=1e-15)

    def test_static_attitude_converges_from_raw_sensor_units(self):
        q_true = normalize_quaternion(np.array([0.9, 0.2, -0.3, 0.25]))
        z_model = observation_model(q_true, 0.45, 0.89)
        for acc_scale, mag_scale in ((1.0, 1.0), (9.81, 48.0)):
            with self.subTest(acc_scale=acc_scale, mag_scale=mag_scale):
                kf = TurboCKF(4, 6, 0.01, hx=observation_model, fx=lambda s, dt: s)
                kf.x = [1.0, 0.0, 0.0, 0.0]
                kf.P = 0.1
                kf.Q = 1e-6
                z = np.r_[acc_scale * z_model[:3], mag_scale * z_model[3:]]
                for _ in range(300):
                    kf.predict_linear_model(np.eye(4))
                    kf.update_paper_ahrs(z, 1e-2, 1e-2)
                    kf.normalize_state_quaternion_backend()
                self.assertLess(_attitude_error_deg(kf.x, q_true), 1.0)

    def test_rejects_zero_or_overflowing_sensor_norms(self):
        good_acc = np.array([0.0, 0.0, 9.81])
        good_mag = np.array([20.0, 0.0, 40.0])
        bad_parts = {
            "zero accelerometer": np.r_[np.zeros(3), good_mag],
            "zero magnetometer": np.r_[good_acc, np.zeros(3)],
            "overflowing accelerometer norm": np.r_[np.full(3, 1e200), good_mag],
            "overflowing magnetometer norm": np.r_[good_acc, np.full(3, 1e200)],
        }
        for label, z in bad_parts.items():
            with self.subTest(label):
                kf = _ahrs_filter()
                with self.assertRaisesRegex(ValueError, "positive finite norms"):
                    kf.update_paper_ahrs(z, sigma_acc2=1e-2, sigma_mag2=1e-2)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from turbo_ckf import TurboCKF
from turbo_ckf.paper_ahrs import (
    magnetic_reference_terms,
    measurement_noise,
    normalize_quaternion,
    observation_model,
    process_noise_from_quaternion,
    transition_matrix_from_gyro,
)


class PaperAlignmentTests(unittest.TestCase):
    def test_transition_matrix_from_gyro(self):
        dt = 0.01
        omega = np.array([0.3, -0.2, 0.1], dtype=float)
        f = transition_matrix_from_gyro(omega, dt)
        self.assertEqual(f.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(f), np.ones(4), atol=1e-12))

    def test_process_noise_from_quaternion(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        qn = process_noise_from_quaternion(q, dt=0.01, gyro_variance=1e-3)
        self.assertEqual(qn.shape, (4, 4))
        self.assertTrue(np.allclose(qn, qn.T, atol=1e-12))
        eigvals = np.linalg.eigvalsh(qn)
        self.assertTrue(np.all(eigvals >= -1e-15))

    def test_observation_model_vectorized_matches_scalar(self):
        q = normalize_quaternion(np.array([0.95, 0.05, -0.1, 0.2], dtype=float))
        q_batch = np.vstack([q, q])
        out0 = observation_model(q, m_n=0.8, m_d=0.6)
        outb = observation_model(q_batch, m_n=0.8, m_d=0.6)
        self.assertEqual(out0.shape, (6,))
        self.assertEqual(outb.shape, (2, 6))
        self.assertTrue(np.allclose(out0, outb[0], atol=1e-12))
        self.assertTrue(np.allclose(out0, outb[1], atol=1e-12))

    def test_dynamic_fk_kckf_matches_ckf_prediction(self):
        dt = 0.01
        gyro_variance = 1e-3
        def hx_dummy(x):
            if x.ndim == 2:
                return np.zeros((x.shape[0], 6), dtype=float)
            return np.zeros(6, dtype=float)

        kckf = TurboCKF(dim_x=4, dim_z=6, dt=dt, hx=hx_dummy, fx=lambda x, dt: x)
        ckf = TurboCKF(dim_x=4, dim_z=6, dt=dt, hx=hx_dummy, fx=lambda x, dt: x)

        x0 = normalize_quaternion(np.array([1.0, 0.02, -0.03, 0.04], dtype=float))
        p0 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.1, 0.1, 0.0],
                [0.0, 0.1, 1.2, 0.0],
                [0.0, 0.0, 0.0, 1.3],
            ],
            dtype=float,
        ) * 1e-2

        for f in (kckf, ckf):
            f.x = x0.copy()
            f.P = p0.copy()

        for step in range(200):
            omega = np.array(
                [
                    0.4 * np.sin(0.01 * step),
                    0.3 * np.cos(0.02 * step),
                    0.2 * np.sin(0.015 * step),
                ],
                dtype=float,
            )
            f = transition_matrix_from_gyro(omega, dt)

            for impl in (kckf, ckf):
                impl.Q = process_noise_from_quaternion(impl.x, dt, gyro_variance)

            kckf.predict_linear_model(f)
            ckf.predict_linear_model_ckf(f)
            kckf.normalize_state_quaternion()
            ckf.normalize_state_quaternion()

        self.assertTrue(np.allclose(kckf.x, ckf.x, atol=1e-10))
        self.assertTrue(np.allclose(kckf.P, ckf.P, atol=1e-10))

    def test_measurement_noise_and_reference_terms(self):
        r = measurement_noise(1e-2, 2e-2)
        self.assertEqual(r.shape, (6, 6))
        self.assertTrue(np.allclose(r[:3, :3], 1e-2 * np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(r[3:, 3:], 2e-2 * np.eye(3), atol=1e-12))

        m_n, m_d = magnetic_reference_terms(np.array([0.0, 0.0, 1.0]), np.array([0.8, 0.0, 0.6]))
        self.assertAlmostEqual(m_n, 0.8, places=12)
        self.assertAlmostEqual(m_d, 0.6, places=12)

    def test_update_paper_ahrs_matches_callback_path(self):
        dt = 0.01
        sigma_acc2 = 1e-2
        sigma_mag2 = 1e-2
        z = np.array([0.0, 0.0, 1.0, 0.8, 0.0, 0.6], dtype=float)
        m_n, m_d = magnetic_reference_terms(z[:3], z[3:])

        ckf_rust = TurboCKF(dim_x=4, dim_z=6, dt=dt, hx=observation_model, fx=lambda x, dt: x)
        ckf_cb = TurboCKF(dim_x=4, dim_z=6, dt=dt, hx=observation_model, fx=lambda x, dt: x)

        x0 = normalize_quaternion(np.array([1.0, 0.04, -0.02, 0.03], dtype=float))
        p0 = 1e-2 * np.array(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.1, 1.2, 0.1, 0.0],
                [0.0, 0.1, 1.1, 0.1],
                [0.0, 0.0, 0.1, 0.9],
            ],
            dtype=float,
        )
        q = 1e-3 * np.eye(4, dtype=float)

        for f in (ckf_rust, ckf_cb):
            f.x = x0.copy()
            f.P = p0.copy()
            f.Q = q.copy()

        fmat = transition_matrix_from_gyro(np.array([0.1, -0.2, 0.05], dtype=float), dt)
        ckf_rust.predict_linear_model(fmat)
        ckf_cb.predict_linear_model(fmat)

        ckf_rust.update_paper_ahrs(z, sigma_acc2=sigma_acc2, sigma_mag2=sigma_mag2)
        ckf_cb.update(z, R=measurement_noise(sigma_acc2, sigma_mag2), hx_args=(m_n, m_d))

        self.assertTrue(np.allclose(ckf_rust.x, ckf_cb.x, atol=1e-10))
        self.assertTrue(np.allclose(ckf_rust.P, ckf_cb.P, atol=1e-10))


if __name__ == "__main__":
    unittest.main()

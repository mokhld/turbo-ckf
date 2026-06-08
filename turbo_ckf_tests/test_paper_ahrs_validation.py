"""Validation error-path coverage for turbo_ckf.paper_ahrs.

The paper_alignment suite checks the numeric output of these AHRS helpers;
this file covers the input guards (wrong vector lengths, non-positive /
non-finite norms, malformed observation-model shapes) they raise on.
"""

import unittest

import numpy as np

from turbo_ckf import (
    magnetic_reference_terms,
    normalize_quaternion,
    observation_model,
    transition_matrix_from_gyro,
)


class NormalizeQuaternionTests(unittest.TestCase):
    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            normalize_quaternion(np.array([1.0, 0.0, 0.0]))

    def test_rejects_zero_norm(self):
        with self.assertRaises(ValueError):
            normalize_quaternion(np.zeros(4))

    def test_rejects_non_finite_norm(self):
        with self.assertRaises(ValueError):
            normalize_quaternion(np.array([np.inf, 0.0, 0.0, 0.0]))

    def test_unit_norm_result(self):
        out = normalize_quaternion(np.array([0.0, 0.0, 3.0, 4.0]))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=12)


class TransitionMatrixTests(unittest.TestCase):
    def test_rejects_wrong_omega_length(self):
        with self.assertRaises(ValueError):
            transition_matrix_from_gyro(np.array([0.1, 0.2]), dt=0.01)


class MagneticReferenceTermsTests(unittest.TestCase):
    def test_rejects_wrong_lengths(self):
        with self.assertRaises(ValueError):
            magnetic_reference_terms(np.zeros(2), np.zeros(3))
        with self.assertRaises(ValueError):
            magnetic_reference_terms(np.zeros(3), np.zeros(2))

    def test_rejects_zero_norms(self):
        with self.assertRaises(ValueError):
            magnetic_reference_terms(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            magnetic_reference_terms(np.array([0.0, 0.0, 1.0]), np.zeros(3))

    def test_returns_normalized_reference(self):
        m_n, m_d = magnetic_reference_terms(
            np.array([0.0, 0.0, 1.0]), np.array([0.8, 0.0, 0.6])
        )
        # m_n^2 + m_d^2 == 1 for unit reference vectors.
        self.assertAlmostEqual(m_n * m_n + m_d * m_d, 1.0, places=12)


class ObservationModelTests(unittest.TestCase):
    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            observation_model(np.zeros((2, 3)), m_n=0.8, m_d=0.6)
        with self.assertRaises(ValueError):
            observation_model(np.zeros((2, 2, 4)), m_n=0.8, m_d=0.6)


if __name__ == "__main__":
    unittest.main()

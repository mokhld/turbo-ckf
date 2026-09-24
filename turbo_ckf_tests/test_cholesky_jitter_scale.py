"""stable_cholesky jitter is relative to each diagonal entry.

The jitter ladder used to be absolute (1e-12 up to 1e-6). Once a diagonal
entry passes about 1.7e10, 1e-6 is less than half an ulp, so every rung
rounds away and a numerically singular covariance raised "unable to compute
stable Cholesky factor". At the other end, a 1e-12 rung is large next to a
small variance and silently changes the filter's answer. Whether a filter
worked, and what NIS it reported, depended on the units of its states.

The scenario: two receiver clocks disciplined by one reference oscillator
share an unknown offset (sigma 1 ms), so their prior covariance is exactly
rank one. A time-interval counter measures their difference with sigma 1 us.
A priori the difference is known, so a reading of 0.5 us has NIS 0.25.
"""

from __future__ import annotations

import unittest

import numpy as np

from turbo_ckf import TurboCKF, TurboSRCKF

UNITS = {"s": 1.0, "us": 1e6, "ns": 1e9}


def fx_static(sigmas, dt):
    return sigmas.copy()


def hx_difference(sigmas):
    return (sigmas[:, 0] - sigmas[:, 1]).reshape(-1, 1)


def _clock_pair(cls, unit: float):
    """The two-clock filter with times expressed in `unit` per second."""
    kf = cls(dim_x=2, dim_z=1, dt=1.0, fx=fx_static, hx=hx_difference)
    kf.x = [0.0, 0.0]
    kf.P = (1e-3 * unit) ** 2 * np.ones((2, 2))
    kf.Q = (1e-12 * unit) ** 2 * np.eye(2)
    kf.R = (1e-6 * unit) ** 2
    return kf


class JitterScaleTests(unittest.TestCase):
    def test_rank_one_prior_in_nanoseconds_is_rescued(self):
        # P = 1e12 * ones: the old ladder added nothing and raised.
        kf = _clock_pair(TurboCKF, UNITS["ns"])
        kf.predict()
        self.assertEqual(kf.jitter_count, 1)
        self.assertAlmostEqual(kf.last_jitter, 1.0, delta=1e-9)  # 1e-12 * 1e12 ns^2
        self.assertTrue(np.all(np.isfinite(kf.P)))

    def test_nis_does_not_depend_on_time_unit(self):
        for cls in (TurboCKF, TurboSRCKF):
            for name, unit in UNITS.items():
                with self.subTest(cls=cls.__name__, unit=name):
                    kf = _clock_pair(cls, unit)
                    for _ in range(3):
                        kf.predict()
                        kf.update(np.array([0.5e-6 * unit]))
                        self.assertAlmostEqual(kf.nis, 0.25, delta=1e-4)
                    np.testing.assert_allclose(np.diag(kf.P) / unit**2, 1e-6, rtol=1e-9)

    def test_small_variance_states_are_not_inflated(self):
        # A near-singular block at 1e12 next to a well-conditioned 1e-6
        # block. Jitter scaled by the mean diagonal would add about 0.5 to
        # the small block; per-entry scaling leaves it at 1e-6.
        kf = TurboCKF(
            dim_x=4,
            dim_z=3,
            dt=1.0,
            fx=fx_static,
            hx=lambda s: np.column_stack([s[:, 0] - s[:, 1], s[:, 2], s[:, 3]]),
        )
        P0 = np.zeros((4, 4))
        P0[:2, :2] = 1e12
        P0[2, 2] = P0[3, 3] = 1e-6
        kf.P = P0
        kf.Q = np.zeros((4, 4))
        kf.predict()
        self.assertEqual(kf.jitter_count, 1)
        np.testing.assert_allclose(np.diag(kf.P), np.diag(P0), rtol=1e-9)

    def test_zero_q_still_factors_on_srckf(self):
        # Q = 0 has no positive diagonal to scale by; it falls back to 1.0.
        kf = TurboSRCKF(dim_x=2, dim_z=1, dt=0.1, fx=fx_static, hx=hx_difference)
        kf.Q = np.zeros((2, 2))
        kf.predict()
        np.testing.assert_allclose(kf.chol_Q, 1e-6 * np.eye(2), rtol=1e-9)

    def test_python_mirror_is_relative(self):
        # Powers of two keep the factorization exactly singular, so the
        # jitter path runs at every scale.
        for scale in (2.0**40, 1.0, 2.0**-40):
            with self.subTest(scale=scale):
                cov = scale * np.ones((2, 2))
                chol = TurboCKF._stable_cholesky(cov)
                np.testing.assert_allclose(chol @ chol.T, cov, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()

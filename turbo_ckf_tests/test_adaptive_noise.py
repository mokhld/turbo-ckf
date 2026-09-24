"""Adaptive noise (Sage-Husa R/Q) estimator coverage.

Acceptance criteria from Session 6:
- enable_adaptive_noise(...) is opt-in; default OFF keeps existing behaviour
- on a CV trajectory where the true R is 4x the user's initial R, adaptive R
  converges to within 25% relative error in <= 500 steps and the
  non-adaptive baseline remains noticeably biased on time-averaged NIS
- P stays symmetric / PSD across 10k steps with adaptive R running
- input validation rejects bad window / alpha / mode
- to_dict / from_dict / copy() round-trip preserves estimator state
- disable_adaptive_noise drops the estimator state and reverts behaviour
"""

from __future__ import annotations

import unittest

import numpy as np

from turbo_ckf import TurboCKF
from turbo_ckf.core import _AdaptiveNoiseEstimator


def fx_cv(x: np.ndarray, dt: float) -> np.ndarray:
    """Vectorized constant-velocity transition for state [pos, vel]."""

    out = x.copy()
    out[:, 0] = x[:, 0] + dt * x[:, 1]
    return out


def hx_pos(x: np.ndarray) -> np.ndarray:
    """Observe position only."""

    return x[:, :1]


def _simulate_cv_trajectory(
    steps: int,
    dt: float,
    true_R_diag: float,
    process_noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``(zs, true_states)`` for a 2-state CV target observing position."""

    rng = np.random.default_rng(seed)
    x_true = np.array([0.0, 1.0], dtype=float)
    zs = np.empty((steps, 1), dtype=float)
    states = np.empty((steps, 2), dtype=float)
    for k in range(steps):
        # True transition: x_pos += dt * x_vel, x_vel += small process noise.
        x_true = np.array(
            [x_true[0] + dt * x_true[1], x_true[1] + rng.normal(0.0, np.sqrt(process_noise))]
        )
        states[k] = x_true
        zs[k, 0] = x_true[0] + rng.normal(0.0, np.sqrt(true_R_diag))
    return zs, states


def _build_filter(initial_R: float) -> TurboCKF:
    kf = TurboCKF(dim_x=2, dim_z=1, dt=0.05, hx=hx_pos, fx=fx_cv)
    kf.x = np.array([0.0, 1.0], dtype=float)
    kf.P = np.eye(2)
    kf.Q = 1e-4 * np.eye(2)
    kf.R = np.array([[initial_R]], dtype=float)
    return kf


def fx_cv4(x: np.ndarray, dt: float) -> np.ndarray:
    """Vectorized 2-D constant-velocity transition for state [px, py, vx, vy]."""

    out = x.copy()
    out[:, :2] = x[:, :2] + dt * x[:, 2:]
    return out


def hx_pos2(x: np.ndarray) -> np.ndarray:
    """Observe both position components."""

    return x[:, :2]


def _build_filter_2d(initial_P: float, initial_R: float) -> TurboCKF:
    """4-state CV filter observing 2-D position (the REVIEW A4 setup)."""

    kf = TurboCKF(dim_x=4, dim_z=2, dt=0.1, hx=hx_pos2, fx=fx_cv4)
    kf.P = initial_P
    kf.Q = 1e-4
    kf.R = initial_R
    return kf


class AdaptiveNoiseOptInTests(unittest.TestCase):
    """Default OFF; methods exist; explicit toggles work cleanly."""

    def test_default_is_off(self):
        kf = _build_filter(initial_R=0.01)
        self.assertIsNone(kf.adaptive_noise_estimator)
        for _ in range(10):
            kf.predict()
            kf.update(np.array([0.0]))
        # R should be exactly what we set — adaptive never ran.
        np.testing.assert_allclose(kf.R, np.array([[0.01]]))

    def test_enable_then_disable_returns_to_off(self):
        kf = _build_filter(initial_R=0.01)
        kf.enable_adaptive_noise(window=5, mode="R", alpha=0.5)
        self.assertIsNotNone(kf.adaptive_noise_estimator)
        kf.disable_adaptive_noise()
        self.assertIsNone(kf.adaptive_noise_estimator)

    def test_invalid_arguments_raise(self):
        kf = _build_filter(initial_R=0.01)
        with self.assertRaises(ValueError):
            kf.enable_adaptive_noise(window=0, mode="R", alpha=0.1)
        with self.assertRaises(ValueError):
            kf.enable_adaptive_noise(window=10, mode="bogus", alpha=0.1)
        with self.assertRaises(ValueError):
            kf.enable_adaptive_noise(window=10, mode="R", alpha=0.0)
        with self.assertRaises(ValueError):
            kf.enable_adaptive_noise(window=10, mode="R", alpha=1.5)


class AdaptiveConvergenceTests(unittest.TestCase):
    """R must track the true measurement noise; baseline must be biased."""

    def test_R_converges_when_initial_is_4x_too_small(self):
        true_R = 0.04  # variance
        initial_R = true_R / 4.0  # filter starts with R one-quarter true
        zs, _ = _simulate_cv_trajectory(
            steps=500, dt=0.05, true_R_diag=true_R, process_noise=1e-5, seed=42
        )

        # Adaptive run.
        kf_adapt = _build_filter(initial_R=initial_R)
        kf_adapt.enable_adaptive_noise(window=30, mode="R", alpha=0.1)
        nis_adapt: list[float] = []
        for k in range(zs.shape[0]):
            kf_adapt.predict()
            kf_adapt.update(zs[k])
            if np.isfinite(kf_adapt.nis):
                nis_adapt.append(kf_adapt.nis)
        final_R_adapt = float(kf_adapt.R[0, 0])
        rel_err = abs(final_R_adapt - true_R) / true_R
        self.assertLess(
            rel_err,
            0.25,
            f"adaptive R = {final_R_adapt:.4f} too far from true {true_R:.4f} "
            f"(relative error {rel_err:.2%})",
        )

        # Baseline run (no adaptation) — R stays at the wrong value.
        kf_base = _build_filter(initial_R=initial_R)
        nis_base: list[float] = []
        for k in range(zs.shape[0]):
            kf_base.predict()
            kf_base.update(zs[k])
            if np.isfinite(kf_base.nis):
                nis_base.append(kf_base.nis)
        self.assertAlmostEqual(float(kf_base.R[0, 0]), initial_R, places=10)

        # For dim_z = 1, chi-squared mean is 1. The baseline filter underestimates
        # R, so its innovation S is too small and NIS inflates well above 1.
        # The adaptive filter should be markedly closer to 1.
        mean_nis_base = float(np.mean(nis_base[-200:]))
        mean_nis_adapt = float(np.mean(nis_adapt[-200:]))
        self.assertLess(
            abs(mean_nis_adapt - 1.0),
            abs(mean_nis_base - 1.0),
            f"adaptive NIS mean {mean_nis_adapt:.2f} should be closer to 1 than "
            f"baseline {mean_nis_base:.2f}",
        )
        # Baseline must be measurably biased — > 2x the chi-sq mean is plenty
        # of margin given R is 1/4 of truth.
        self.assertGreater(mean_nis_base, 2.0)


class AdaptiveStabilityTests(unittest.TestCase):
    """P stays PSD across 10k steps with adaptive running."""

    def test_p_stays_psd_with_adaptive_R_over_10k_steps(self):
        kf = _build_filter(initial_R=0.01)
        kf.enable_adaptive_noise(window=20, mode="R", alpha=0.05)
        rng = np.random.default_rng(123)
        true_pos = 0.0
        true_vel = 1.0
        max_asym = 0.0
        min_eig = np.inf
        min_R = np.inf
        for _ in range(10_000):
            true_pos += kf.dt * true_vel
            kf.predict()
            z = np.array([true_pos + rng.normal(0.0, np.sqrt(0.04))], dtype=float)
            kf.update(z)
            asym = float(np.max(np.abs(kf.P - kf.P.T)))
            if asym > max_asym:
                max_asym = asym
            eig = float(np.min(np.linalg.eigvalsh(0.5 * (kf.P + kf.P.T))))
            if eig < min_eig:
                min_eig = eig
            if kf.R[0, 0] < min_R:
                min_R = float(kf.R[0, 0])

        self.assertLess(max_asym, 1e-10, "P drifted asymmetric under adaptive R")
        self.assertGreater(min_eig, -1e-10, "P drifted non-PSD under adaptive R")
        self.assertGreater(min_R, 0.0, "adaptive R diagonal went non-positive")

    def test_p_stays_psd_with_mode_both(self):
        # Q adaptation is the delicate channel; ensure it doesn't blow up
        # over a long run when alpha is kept conservatively small.
        kf = _build_filter(initial_R=0.01)
        kf.enable_adaptive_noise(window=50, mode="both", alpha=0.02)
        rng = np.random.default_rng(7)
        true_pos = 0.0
        true_vel = 1.0
        for _ in range(2_000):
            true_pos += kf.dt * true_vel
            kf.predict()
            z = np.array([true_pos + rng.normal(0.0, np.sqrt(0.04))], dtype=float)
            kf.update(z)
        eig = float(np.min(np.linalg.eigvalsh(0.5 * (kf.P + kf.P.T))))
        self.assertGreater(eig, -1e-10)
        # Adaptive Q diagonal should remain positive (diagonal_floor enforced).
        self.assertTrue(np.all(np.diag(kf.Q) > 0.0))


class AdaptiveSerializationTests(unittest.TestCase):
    """copy() and to_dict() round-trip preserve estimator state."""

    def test_copy_carries_adaptive_estimator(self):
        kf = _build_filter(initial_R=0.01)
        kf.enable_adaptive_noise(window=5, mode="R", alpha=0.2)
        for _ in range(20):
            kf.predict()
            kf.update(np.array([0.0]))
        clone = kf.copy()
        self.assertIsNotNone(clone.adaptive_noise_estimator)
        np.testing.assert_allclose(
            kf.adaptive_noise_estimator.estimate_R(),
            clone.adaptive_noise_estimator.estimate_R(),
        )
        self.assertEqual(
            kf.adaptive_noise_estimator.count,
            clone.adaptive_noise_estimator.count,
        )

    def test_to_dict_from_dict_round_trip_carries_adaptive(self):
        kf = _build_filter(initial_R=0.01)
        kf.enable_adaptive_noise(window=5, mode="both", alpha=0.1)
        for _ in range(15):
            kf.predict()
            kf.update(np.array([0.0]))
        snap = kf.to_dict()
        self.assertIn("adaptive", snap)
        restored = TurboCKF.from_dict(snap, hx=hx_pos, fx=fx_cv)
        self.assertIsNotNone(restored.adaptive_noise_estimator)
        np.testing.assert_allclose(
            kf.adaptive_noise_estimator.estimate_R(),
            restored.adaptive_noise_estimator.estimate_R(),
        )
        np.testing.assert_allclose(
            kf.adaptive_noise_estimator.estimate_Q(),
            restored.adaptive_noise_estimator.estimate_Q(),
        )
        self.assertEqual(
            kf.adaptive_noise_estimator.mode,
            restored.adaptive_noise_estimator.mode,
        )

    def test_to_dict_omits_adaptive_when_disabled(self):
        kf = _build_filter(initial_R=0.01)
        snap = kf.to_dict()
        self.assertNotIn("adaptive", snap)


class AdaptiveEstimatorUnitTests(unittest.TestCase):
    """Direct tests of the estimator math, no filter loop."""

    def test_warmup_window_blocks_writeback(self):
        est = _AdaptiveNoiseEstimator(window=5, mode="R", alpha=0.5, dim_x=2, dim_z=1)
        y = np.array([0.1])
        S = np.array([[0.05]])
        R = np.array([[0.02]])
        K = np.zeros((2, 1))
        for i in range(4):
            new_R, new_Q = est.step(y, S, R, K)
            self.assertIsNone(new_R, f"R should be None at step {i + 1} (< window=5)")
            self.assertIsNone(new_Q)
        new_R, new_Q = est.step(y, S, R, K)
        self.assertIsNotNone(new_R, "R should be written back once count == window")
        self.assertIsNone(new_Q, "Q-channel disabled by mode='R'")

    def test_diagonal_floor_keeps_R_positive(self):
        # A zero R and a zero innovation give a zero contribution; the
        # write-back must still be positive-definite. Checks eigenvalues, not
        # just the diagonal: a positive diagonal is what let an indefinite R
        # through before (REVIEW A4).
        est = _AdaptiveNoiseEstimator(
            window=1, mode="R", alpha=1.0, dim_x=2, dim_z=2, diagonal_floor=1e-9
        )
        y = np.zeros(2)
        S = 10.0 * np.eye(2)
        R = np.zeros((2, 2))
        K = np.zeros((2, 2))
        new_R, _ = est.step(y, S, R, K)
        self.assertIsNotNone(new_R)
        self.assertTrue(np.all(np.linalg.eigvalsh(new_R) > 0.0))

    def test_clamp_pd_projects_indefinite_input(self):
        # REVIEW A4's floored estimate, eigenvalues -0.0155 and +0.0155. A
        # diagonal-only floor leaves it indefinite.
        est = _AdaptiveNoiseEstimator(
            window=1, mode="R", alpha=1.0, dim_x=2, dim_z=2, diagonal_floor=1e-9
        )
        indefinite = np.array([[1e-12, 0.0155], [0.0155, 1e-12]])
        out = est._clamp_pd(indefinite)
        np.testing.assert_array_equal(out, out.T)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(out), [1e-9, 0.0155 + 1e-12], rtol=1e-6
        )
        np.linalg.cholesky(out)  # raises LinAlgError unless positive-definite

    def test_clamp_pd_leaves_pd_input_unchanged(self):
        est = _AdaptiveNoiseEstimator(window=1, mode="R", alpha=1.0, dim_x=2, dim_z=2)
        pd = np.array([[0.3, 0.1], [0.1, 0.2]])
        np.testing.assert_allclose(est._clamp_pd(pd), pd, rtol=1e-12)

    def test_R_contribution_is_residual_form(self):
        # For a linear h, one step with alpha=1 must equal
        # e e^T + H P_post H^T with e = z - H x_post (Akhlaghi et al. 2017),
        # here in a case where the innovation form y y^T + R - S is indefinite.
        rng = np.random.default_rng(3)
        H = rng.normal(size=(2, 4))
        A = rng.normal(size=(4, 4))
        P = A @ A.T + 0.1 * np.eye(4)
        R = np.array([[0.5, 0.1], [0.1, 0.3]])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y = np.array([0.1, -0.05])
        self.assertLess(np.linalg.eigvalsh(np.outer(y, y) + R - S)[0], 0.0)

        e = y - H @ K @ y
        P_post = P - K @ S @ K.T
        expected = np.outer(e, e) + H @ P_post @ H.T

        est = _AdaptiveNoiseEstimator(window=1, mode="R", alpha=1.0, dim_x=4, dim_z=2)
        new_R, _ = est.step(y, S, R, K)
        np.testing.assert_allclose(est.estimate_R(), expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(new_R, expected, rtol=1e-10, atol=1e-12)

    def test_singular_S_uses_pseudo_inverse(self):
        # The backend update accepts an exactly singular S (pinv fallback);
        # the estimator must not raise there. With S^+ in place of S^-1:
        # e = R S^+ y = [0.1, 0] and R - R S^+ R = diag(0.25, 0).
        est = _AdaptiveNoiseEstimator(
            window=1, mode="R", alpha=1.0, dim_x=2, dim_z=2, diagonal_floor=1e-9
        )
        S = np.diag([1.0, 0.0])
        R = np.diag([0.5, 0.0])
        new_R, _ = est.step(np.array([0.2, 0.0]), S, R, np.zeros((2, 2)))
        np.testing.assert_allclose(est.estimate_R(), np.diag([0.26, 0.0]), atol=1e-15)
        np.testing.assert_allclose(new_R, np.diag([0.26, 1e-9]), rtol=1e-9, atol=1e-15)


class AdaptiveOverestimatedStartTests(unittest.TestCase):
    """REVIEW A4: dim_z=2 with R or P starting overestimated.

    Guards against two failure modes of the innovation-based estimate
    ``y y^T + R - S``, which is negative while P overstates the real error:
    an indefinite write-back that makes the filter raise "unable to compute
    stable Cholesky factor", and, with the write-back projected but the
    estimate unchanged, R collapsing to the floor and the filter diverging.
    """

    # (initial_P, initial_R, true_R, alpha): the REVIEW A4 crash cases plus
    # its repro snippet, which uses the default alpha.
    SCENARIOS = (
        (100.0, 1.0, 0.01, 0.3),
        (10.0, 1.0, 0.25, 0.3),
        (1.0, 4.0, 0.25, 0.05),
        (1.0, 4.0, 0.25, 0.3),
    )
    SEEDS = range(20)
    STEPS = 500
    TAIL = 200

    @classmethod
    def setUpClass(cls):
        cls.results = {
            (scenario, seed): cls._run(*scenario, seed)
            for scenario in cls.SCENARIOS
            for seed in cls.SEEDS
        }

    @classmethod
    def _run(cls, initial_P, initial_R, true_R, alpha, seed):
        # Stationary target at the origin, so z is pure measurement noise.
        kf = _build_filter_2d(initial_P, initial_R)
        kf.enable_adaptive_noise(alpha=alpha)
        rng = np.random.default_rng(seed)
        symmetric = True
        min_eig = np.inf
        R_sum = np.zeros((2, 2))
        try:
            for k in range(cls.STEPS):
                kf.predict()
                kf.update(rng.normal(0.0, np.sqrt(true_R), 2))
                symmetric = symmetric and np.array_equal(kf.R, kf.R.T)
                min_eig = min(min_eig, float(np.linalg.eigvalsh(kf.R)[0]))
                if k >= cls.STEPS - cls.TAIL:
                    R_sum += kf.R
        except RuntimeError as exc:
            return {"error": str(exc)}
        truth = true_R * np.eye(2)
        return {
            "error": None,
            "symmetric": symmetric,
            "min_eig": min_eig,
            "R_err": float(np.linalg.norm(R_sum / cls.TAIL - truth, 2) / true_R),
            "x_err": float(np.max(np.abs(kf.x[:2])) / np.sqrt(true_R)),
        }

    def test_scenarios_complete_with_symmetric_pd_R_every_step(self):
        crashed = {
            key: res["error"] for key, res in self.results.items() if res["error"]
        }
        self.assertEqual(
            len(crashed),
            0,
            f"{len(crashed)} of {len(self.results)} runs raised: {crashed}",
        )
        for key, res in self.results.items():
            self.assertTrue(res["symmetric"], f"R lost exact symmetry in run {key}")
            self.assertGreater(res["min_eig"], 0.0, f"R went non-PD in run {key}")

    def test_R_converges_toward_truth_without_diverging(self):
        # Starting relative errors are 99, 3 and 15. Measured over these
        # 80 runs, R averaged over the last 200 steps is within 29% of the
        # truth (spectral norm, relative; median 13-18%) and the final
        # position error is under 0.9 sigma. Single-step R is noisier (up to
        # ~3x off at alpha=0.3), hence the time average. A diverging filter
        # misses both bounds by orders of magnitude (R error 10 to 1e6,
        # position error up to ~1000 sigma).
        for key, res in self.results.items():
            self.assertIsNone(res["error"], f"run {key} raised")
            self.assertLess(
                res["R_err"],
                0.5,
                f"run {key}: time-averaged R off by {res['R_err']:.1%}",
            )
            self.assertLess(
                res["x_err"], 5.0, f"run {key}: position error {res['x_err']:.1f} sigma"
            )


class AdaptiveROverrideTests(unittest.TestCase):
    """Adaptation must use the R each update applied, not the stored self.R."""

    def test_run_with_Rs_override_estimates_true_R(self):
        true_R = 0.25
        # Stored R is 4x the truth; every step overrides it with the truth.
        kf = _build_filter_2d(initial_P=1.0, initial_R=1.0)
        kf.enable_adaptive_noise(alpha=0.02)
        zs = np.random.default_rng(0).normal(0.0, np.sqrt(true_R), (500, 2))
        kf.run(zs, Rs=true_R)
        # Measured over seeds 0-19: final relative error median 17%, max 31%.
        rel_err = float(np.linalg.norm(kf.R - true_R * np.eye(2), 2) / true_R)
        self.assertLess(rel_err, 0.5, f"adaptive R = {kf.R} vs true {true_R}")


if __name__ == "__main__":
    unittest.main()

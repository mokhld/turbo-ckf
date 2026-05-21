"""Tests for the Square-Root Cubature Kalman Filter (TurboSRCKF).

Three pillars (matching the Session 4 acceptance criteria):

1. Parity vs the standard TurboCKF on a well-conditioned linear-Gaussian
   constant-velocity trajectory. State and covariance should agree to ~1e-8.

2. On an ill-conditioned trajectory, TurboCKF.jitter_count grows (its
   predict-side stable_cholesky on P has to add diagonal jitter to recover
   PD) while TurboSRCKF.jitter_count stays at zero (it never re-decomposes
   P in the filter loop).

3. Basic API sanity: snapshot shapes, chol_P · chol_P^T == P, skipped
   updates, validation errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from turbo_ckf import TurboCKF, TurboSRCKF


def _cv_fx(sigmas, dt):
    f = np.eye(4)
    f[0, 2] = dt
    f[1, 3] = dt
    return sigmas @ f.T


def _pos_hx(sigmas):
    return sigmas[:, :2]


def _make_pair(dim_x, dim_z, dt, hx, fx, x0, P0, Q, R):
    ckf = TurboCKF(dim_x, dim_z, dt, hx=hx, fx=fx)
    ckf.x = x0.copy()
    ckf.P = P0.copy()
    ckf.Q = Q.copy()
    ckf.R = R.copy()

    srckf = TurboSRCKF(dim_x, dim_z, dt, hx=hx, fx=fx)
    srckf.x = x0.copy()
    srckf.P = P0.copy()
    srckf.Q = Q.copy()
    srckf.R = R.copy()
    return ckf, srckf


# ---------------------------------------------------------------------------
# Pillar 1: parity on a well-conditioned trajectory
# ---------------------------------------------------------------------------

def test_srckf_matches_ckf_on_well_conditioned_cv():
    """60-step linear-Gaussian CV filter — TurboSRCKF and TurboCKF should agree
    to ~1e-8 on every step (state and covariance)."""

    rng = np.random.default_rng(0)
    dt = 0.1

    x0 = np.array([0.0, 0.0, 1.0, 0.5])
    P0 = np.diag([1.0, 1.0, 0.5, 0.5])
    Q = np.eye(4) * 0.01
    R = np.eye(2) * 0.1

    ckf, srckf = _make_pair(4, 2, dt, _pos_hx, _cv_fx, x0, P0, Q, R)

    # Simulate a noisy CV trajectory.
    F_true = np.eye(4)
    F_true[0, 2] = dt
    F_true[1, 3] = dt
    truth = x0.copy()
    zs = []
    for _ in range(60):
        truth = F_true @ truth + rng.multivariate_normal(np.zeros(4), Q)
        zs.append(truth[:2] + rng.multivariate_normal(np.zeros(2), R))

    max_x_diff = 0.0
    max_P_diff = 0.0
    for z in zs:
        ckf.predict()
        ckf.update(z)
        srckf.predict()
        srckf.update(z)
        max_x_diff = max(max_x_diff, np.max(np.abs(ckf.x - srckf.x)))
        max_P_diff = max(max_P_diff, np.max(np.abs(ckf.P - srckf.P)))

    assert max_x_diff < 1e-8, f"max |x| diff = {max_x_diff}"
    assert max_P_diff < 1e-8, f"max |P| diff = {max_P_diff}"
    assert ckf.jitter_count == 0
    assert srckf.jitter_count == 0
    assert srckf.downdate_fallback_count == 0


def test_srckf_matches_ckf_on_pure_predict_chain():
    """Predict-only parity: 200 nonlinear predict steps with no updates.

    The cubature point regeneration is the heart of the silent-jitter risk
    in TurboCKF; checking that SR-CKF tracks the same prior here verifies
    the QR-based predict path matches the chol-based one in arithmetic.
    """

    dt = 0.05
    x0 = np.array([1.0, 2.0, 0.3, -0.1])
    P0 = np.diag([0.4, 0.4, 0.05, 0.05])
    Q = np.eye(4) * 1e-3
    R = np.eye(2)  # unused

    ckf, srckf = _make_pair(4, 2, dt, _pos_hx, _cv_fx, x0, P0, Q, R)

    for _ in range(200):
        ckf.predict()
        srckf.predict()

    assert np.max(np.abs(ckf.x - srckf.x)) < 1e-8
    assert np.max(np.abs(ckf.P - srckf.P)) < 1e-8


# ---------------------------------------------------------------------------
# Pillar 2: SR-CKF stays clean on an ill-conditioned trajectory
# ---------------------------------------------------------------------------

def test_srckf_jitter_count_stays_zero_on_ill_conditioned_run():
    """On a near-rank-deficient initial P (eigenvalue ratio ~1e18) with tiny
    R, TurboCKF must add diagonal jitter via stable_cholesky to keep P PD,
    while TurboSRCKF — which never re-decomposes P during the filter loop —
    reports zero jitter.

    The acceptance criterion is the inequality (CKF > 0, SR-CKF == 0), not
    the magnitudes. The point is that silent jitter is impossible on the
    square-root path in steady state.
    """

    rng = np.random.default_rng(5)

    # Rotate a near-rank-deficient diagonal P0 so it isn't aligned to axes.
    U = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    eig = np.array([1e6, 1e6, 1e-12, 1e-12])
    P0 = U @ np.diag(eig) @ U.T
    P0 = 0.5 * (P0 + P0.T)

    dt = 0.001
    x0 = np.zeros(4)
    Q = np.eye(4) * 1e-14
    R = np.eye(2) * 1e-10

    ckf, srckf = _make_pair(4, 2, dt, _pos_hx, _cv_fx, x0, P0, Q, R)

    # Discard the one-shot Cholesky-at-seeding cost from SR-CKF's counters —
    # we want to measure per-step jitter, not the cost of accepting the
    # user's degenerate seed. TurboCKF doesn't have a hook for this, but
    # its own seed-time chol happens lazily on the first predict, so its
    # counter naturally starts at zero too.
    srckf.reset_jitter_counters()
    assert srckf.jitter_count == 0
    assert ckf.jitter_count == 0

    for _ in range(5000):
        ckf.predict()
        ckf.update(np.zeros(2))
        srckf.predict()
        srckf.update(np.zeros(2))

    assert ckf.jitter_count > 0, (
        f"TurboCKF didn't trip stable_cholesky on this trajectory "
        f"(jitter_count={ckf.jitter_count}); the ill-conditioning isn't "
        f"actually exercising the silent-jitter path. Tighten the test."
    )
    assert srckf.jitter_count == 0, (
        f"TurboSRCKF reported jitter on the filter loop "
        f"(jitter_count={srckf.jitter_count}). Either the rank-1 downdate "
        f"fell back to a fresh Cholesky (downdate_fallback_count="
        f"{srckf.downdate_fallback_count}) or seed-time jitter leaked into "
        f"the per-step counter."
    )


# ---------------------------------------------------------------------------
# Pillar 3: API / snapshot sanity
# ---------------------------------------------------------------------------

def test_srckf_snapshot_shapes_and_factor_consistency():
    """chol_P is lower-triangular and reconstructs P; covariance attributes
    have the expected shapes after predict+update."""

    dt = 0.05
    x0 = np.array([0.1, -0.2, 0.3, 0.4])
    P0 = np.diag([0.5, 0.5, 0.1, 0.1])
    Q = np.eye(4) * 0.01
    R = np.eye(2) * 0.05

    kf = TurboSRCKF(4, 2, dt, hx=_pos_hx, fx=_cv_fx)
    kf.x = x0.copy()
    kf.P = P0.copy()
    kf.Q = Q.copy()
    kf.R = R.copy()

    kf.predict()
    kf.update(np.array([0.05, -0.1]))

    assert kf.x.shape == (4,)
    assert kf.P.shape == (4, 4)
    assert kf.chol_P.shape == (4, 4)
    assert kf.S.shape == (2, 2)
    assert kf.S_innov.shape == (2, 2)
    assert kf.K.shape == (4, 2)

    # chol_P is lower-triangular and reconstructs P (the snapshot's P is
    # derived from chol_P · chol_P^T inside Rust, so this checks both).
    upper = kf.chol_P - np.tril(kf.chol_P)
    assert np.max(np.abs(upper)) < 1e-12
    assert np.allclose(kf.chol_P @ kf.chol_P.T, kf.P, atol=1e-12)

    # P_post is symmetric PSD.
    eigvals = np.linalg.eigvalsh(0.5 * (kf.P_post + kf.P_post.T))
    assert eigvals.min() > -1e-10


def test_srckf_skipped_update_clears_diagnostics():
    """update(z=None) leaves no stale innovation values around."""

    dt = 0.1
    kf = TurboSRCKF(4, 2, dt, hx=_pos_hx, fx=_cv_fx)
    kf.x = np.zeros(4)
    kf.P = np.eye(4)
    kf.Q = np.eye(4) * 0.01
    kf.R = np.eye(2) * 0.1

    kf.predict()
    kf.update(np.array([0.1, 0.2]))
    assert np.isfinite(kf.log_likelihood)
    assert np.isfinite(kf.nis)

    kf.predict()
    kf.update(None)
    assert not np.isfinite(kf.log_likelihood)
    assert not np.isfinite(kf.nis)
    assert np.all(np.isnan(kf.z))


def test_srckf_rejects_invalid_construction():
    """Constructor validates dim and dt the same way TurboCKF does."""

    with pytest.raises(ValueError):
        TurboSRCKF(0, 2, 0.1, hx=_pos_hx, fx=_cv_fx)
    with pytest.raises(ValueError):
        TurboSRCKF(4, 0, 0.1, hx=_pos_hx, fx=_cv_fx)
    with pytest.raises(ValueError):
        TurboSRCKF(4, 2, float("nan"), hx=_pos_hx, fx=_cv_fx)


def test_srckf_reset_returns_to_defaults():
    """reset() puts the filter back in a known state."""

    dt = 0.1
    kf = TurboSRCKF(4, 2, dt, hx=_pos_hx, fx=_cv_fx)
    kf.x = np.array([1.0, 2.0, 3.0, 4.0])
    kf.P = np.diag([2.0, 2.0, 2.0, 2.0])
    kf.Q = np.eye(4) * 0.01
    kf.R = np.eye(2) * 0.05
    kf.predict()
    kf.update(np.array([0.5, 0.5]))
    assert kf.jitter_count == 0  # ought to be true; mostly a sanity check

    kf.reset()
    assert np.allclose(kf.x, 0.0)
    assert np.allclose(kf.P, np.eye(4))
    assert not np.isfinite(kf.log_likelihood)
    assert kf.jitter_count == 0
    assert kf.downdate_fallback_count == 0


def test_srckf_gate_chi_square():
    """gate() returns True for innovations within the chi-square threshold
    and False before any update has happened."""

    dt = 0.1
    kf = TurboSRCKF(4, 2, dt, hx=_pos_hx, fx=_cv_fx)
    kf.x = np.zeros(4)
    kf.P = np.eye(4)
    kf.Q = np.eye(4) * 0.01
    kf.R = np.eye(2) * 0.1

    # No update yet -> NIS is NaN -> gate must return False.
    assert kf.gate(5.99) is False

    kf.predict()
    kf.update(np.array([0.01, 0.01]))  # tiny innovation, well within gate
    assert kf.gate(5.99) is True
    assert kf.gate(1e-6) is False

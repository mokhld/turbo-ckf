"""TurboCKF stays accurate when the state is far from zero.

TurboCKF builds its cubature covariances from deviations about the mean,
which do not depend on where the state sits. The uncentred form
``E[x x^T] - mean mean^T`` does: with a position near an ECEF coordinate
(6.4e6 m) both terms are about 4.1e13, where float64 values are 0.0078
apart, so a covariance below about 1 m^2 keeps at most two correct digits.
On the 1-D constant-velocity case below that form raised "unable to compute
stable Cholesky factor" at step 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from turbo_ckf import TurboCKF, TurboSRCKF
from turbo_ckf.paper_ahrs import (
    normalize_quaternion,
    observation_model,
    process_noise_from_quaternion,
    transition_matrix_from_gyro,
)

OFFSET = 6.4e6  # about the size of an ECEF coordinate in metres
STEPS = 200
F_CV = np.array([[1.0, 1.0], [0.0, 1.0]])

# Run at OFFSET vs the same run at the origin. Near 6.4e6 float64 values are
# 2**-30 ~ 9.3e-10 apart, so each measurement, sigma point and stored
# position is rounded by up to half of that. Measured worst cases over 200
# steps: 2.0e-9 in position (about 2 ulp), 7.4e-10 in velocity, and 5.3e-8
# relative on every P entry (1.2e-8 once P has settled near 4e-3). TurboSRCKF
# shows the same differences, so they come from rounding the inputs, not
# from the filter. The limits give 5-10x headroom. The uncentred form had an
# absolute error around 0.0078 in P, larger than the settled P itself.
X_ATOL = 1e-8
P_RTOL = 5e-7

# TurboCKF vs TurboSRCKF, both at OFFSET. Measured: positions bitwise equal,
# velocity within 5e-16, P within 2.8e-14. A few ulp of 6.4e6 is allowed on x.
X_ATOL_SR = 5e-9
P_ATOL_SR = 1e-12


def _fx(sigmas, dt):
    out = sigmas.copy()
    out[:, 0] = sigmas[:, 0] + dt * sigmas[:, 1]
    return out


def _hx(sigmas):
    return sigmas[:, :1]


def _predict_callback(kf):
    kf.predict()


def _predict_linear_ckf(kf):
    kf.predict_linear_model_ckf(F_CV)


PREDICT_PATHS = pytest.mark.parametrize(
    "predict", [_predict_callback, _predict_linear_ckf], ids=["callback", "linear_ckf"]
)


def _run_cv(cls, offset, predict):
    """1-D CV filter measuring position = offset + N(0, 0.1) for STEPS steps.

    Returns the state relative to ``[offset, 0]`` and P after every update.
    Subtracting the offset is exact (Sterbenz), so the returned deviations
    are the stored state, not a rounded copy of it.
    """

    kf = cls(2, 1, 1.0, hx=_hx, fx=_fx)
    kf.x = [offset, 0.0]
    kf.P = np.diag([100.0, 1.0])
    kf.Q = np.diag([1e-4, 1e-4])
    kf.R = 1e-2
    noise = np.random.default_rng(0).normal(0.0, 0.1, STEPS)
    xs, ps = [], []
    for v in noise:
        predict(kf)
        kf.update([offset + v])
        xs.append(kf.x - [offset, 0.0])
        ps.append(kf.P.copy())
    return np.array(xs), np.array(ps)


@PREDICT_PATHS
def test_ckf_at_ecef_offset_matches_run_at_origin(predict):
    xs, ps = _run_cv(TurboCKF, OFFSET, predict)
    xs0, ps0 = _run_cv(TurboCKF, 0.0, predict)
    np.testing.assert_allclose(xs, xs0, rtol=0.0, atol=X_ATOL)
    np.testing.assert_allclose(ps, ps0, rtol=P_RTOL, atol=0.0)


@PREDICT_PATHS
def test_ckf_at_ecef_offset_matches_srckf(predict):
    xs, ps = _run_cv(TurboCKF, OFFSET, predict)
    xs_sr, ps_sr = _run_cv(TurboSRCKF, OFFSET, _predict_callback)
    np.testing.assert_allclose(xs, xs_sr, rtol=0.0, atol=X_ATOL_SR)
    np.testing.assert_allclose(ps, ps_sr, rtol=0.0, atol=P_ATOL_SR)


def _run_paper_ahrs():
    """50 gyro predicts plus update_paper_ahrs steps from a fixed seed."""

    dt = 0.01
    rng = np.random.default_rng(3)
    kf = TurboCKF(4, 6, dt, hx=observation_model, fx=lambda s, dt: s)
    kf.x = normalize_quaternion([1.0, 0.04, -0.02, 0.03])
    kf.P = 1e-2 * np.eye(4)
    truth = normalize_quaternion([0.97, 0.12, -0.08, 0.2])
    f = transition_matrix_from_gyro([0.3, -0.2, 0.1], dt)
    for _ in range(50):
        truth = normalize_quaternion(f @ truth)
        kf.Q = process_noise_from_quaternion(kf.x, dt, 1e-3) + 1e-9 * np.eye(4)
        kf.predict_linear_model(f)
        z = observation_model(truth, m_n=0.8, m_d=0.6) + rng.normal(0.0, 0.05, 6)
        z[:3] *= 9.81  # raw accelerometer units
        z[3:] *= 50.0  # raw magnetometer units
        kf.update_paper_ahrs(z, sigma_acc2=1e-2, sigma_mag2=1e-2)
        kf.normalize_state_quaternion()
    return kf


# _run_paper_ahrs() output from the uncentred implementation (v0.8.0).
AHRS_X = [0.9490166259495053, 0.19830175896004257, -0.11380380781248879, 0.21700817816881415]
AHRS_P = [
    [3.1796656092358466e-05, -1.3685484960313127e-05, -1.7310883185211582e-06, -2.0151291889389873e-05],
    [-1.3685484960313127e-05, 5.6217533975601685e-05, 2.338755326235039e-06, 3.7962896223561645e-05],
    [-1.7310883185211582e-06, 2.338755326235039e-06, 2.6372525949545286e-05, 6.736069572787965e-06],
    [-2.0151291889389873e-05, 3.7962896223561645e-05, 6.736069572787965e-06, 8.912848593640759e-05],
]
AHRS_NIS = 1.9237568520965653


def test_update_paper_ahrs_unchanged_by_centred_moments():
    """Quaternions are unit scale, so centring changes nothing measurable."""

    kf = _run_paper_ahrs()
    # Measured change: 5.6e-16 on x, 5.5e-18 on P (entries up to 8.9e-5) and
    # 1.3e-14 relative on NIS.
    np.testing.assert_allclose(kf.x, AHRS_X, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(kf.P, AHRS_P, rtol=0.0, atol=1e-12 * np.abs(AHRS_P).max())
    assert kf.nis == pytest.approx(AHRS_NIS, rel=1e-12)


def _raise(*_args):
    raise RuntimeError("callback failed")


@pytest.mark.parametrize("step", ["predict", "update"])
def test_raising_callback_leaves_jitter_counters_unchanged(step):
    kf = TurboCKF(2, 1, 1.0, hx=_hx, fx=_fx)
    # diag(1, 0) is only semi-definite: the cubature points need diagonal
    # jitter before the callback is called.
    kf.P = np.diag([1.0, 0.0])
    with pytest.raises(RuntimeError, match="callback failed"):
        if step == "predict":
            kf.predict(fx=_raise)
        else:
            kf.update([0.0], hx=_raise)
    kf.update(None)  # a skipped update re-reads the backend diagnostics
    assert (kf.jitter_count, kf.last_jitter, kf.max_jitter) == (0, 0.0, 0.0)

    # Control: the same step with a working callback does record jitter.
    if step == "predict":
        kf.predict()
    else:
        kf.update([0.0])
    assert kf.jitter_count == 1
    assert kf.last_jitter > 0.0


def test_nan_innovation_distance_is_not_reported_as_zero():
    """A NaN NIS must stay NaN; NIS 0 would pass every chi-square gate."""

    kf = TurboCKF(2, 1, 1.0, hx=lambda s: np.zeros((s.shape[0], 1)), fx=_fx)
    # hx is constant, so S == R. A subnormal R makes S^-1 overflow to inf,
    # and with y == 0 the distance y^T S^-1 y is 0 * inf = NaN.
    kf.R = 1e-320
    kf.update([0.0])
    assert np.isnan(kf.nis)
    assert np.isnan(kf.mahalanobis)
    assert not kf.gate(1e9)

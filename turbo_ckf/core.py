"""Turbo CKF wrapper over the required Rust backend."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

try:
    from . import _rust  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "turbo_ckf Rust extension is required. Build/install with `maturin develop` before importing turbo_ckf."
    ) from exc


_STANDARD_MODELS = tuple(_rust.CubatureKalmanFilter.supported_standard_models())
_STANDARD_LAYOUTS = tuple(_rust.CubatureKalmanFilter.supported_standard_layouts())


def _validate_standard_layout(layout: str) -> None:
    name = str(layout)
    if name not in _STANDARD_LAYOUTS:
        choices = ", ".join(repr(m) for m in _STANDARD_LAYOUTS)
        raise ValueError(f"unsupported layout {name!r}; expected one of {choices}")


_ADAPTIVE_MODES = ("R", "Q", "both")


def _coerce_state_vector(value: npt.ArrayLike, size: int, name: str) -> Vector:
    """Coerce a state-vector assignment to a float64 1-D array of ``size``.

    Accepts lists/tuples, integer arrays, and ``(size, 1)`` / ``(1, size)``
    column/row vectors. Rejects wrong sizes and non-finite values with an
    error that names the attribute, so mistakes surface at assignment time
    instead of as a pyo3 conversion error inside the next predict().
    """

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be array-like of floats; got {type(value).__name__}"
        ) from exc
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1 or arr.shape[0] != size:
        raise ValueError(
            f"{name} must be a length-{size} vector (shape ({size},), "
            f"({size}, 1), or (1, {size})); got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values; got {arr!r}")
    return np.ascontiguousarray(arr)


def _coerce_square_matrix(value: npt.ArrayLike, size: int, name: str) -> Matrix:
    """Coerce a covariance-style assignment to a float64 ``(size, size)`` array.

    Accepts a scalar (``s * I``), a length-``size`` 1-D array (diagonal), or a
    full ``(size, size)`` matrix — in any dtype/nested-list spelling. Rejects
    wrong shapes and non-finite values with an error that names the attribute.
    """

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a scalar or array-like of floats; got {type(value).__name__}"
        ) from exc
    if arr.ndim == 0:
        arr = float(arr) * np.eye(size, dtype=float)
    elif arr.ndim == 1:
        if arr.shape[0] != size:
            raise ValueError(
                f"1-D {name} is interpreted as a diagonal and must have "
                f"length {size}; got shape {arr.shape}"
            )
        arr = np.diag(arr)
    elif arr.shape != (size, size):
        raise ValueError(
            f"{name} must be a scalar, a length-{size} diagonal, or a "
            f"({size}, {size}) matrix; got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr)


class FilterRun(NamedTuple):
    """Stacked per-step outputs of :meth:`TurboCKF.run` / :meth:`TurboSRCKF.run`.

    ``log_likelihoods`` and ``nis`` are NaN at steps where the measurement
    was missing (the update was skipped); ``missing`` is the boolean mask of
    those steps.
    """

    xs: Matrix
    Ps: npt.NDArray[np.float64]
    x_priors: Matrix
    P_priors: npt.NDArray[np.float64]
    log_likelihoods: Vector
    nis: Vector
    missing: npt.NDArray[np.bool_]


def _normalize_run_measurements(
    zs: Any, dim_z: int, nan_means_missing: bool
) -> list[Vector | None]:
    """Validate/normalize a measurement sequence for run() up front, so shape
    errors raise before the filter has advanced a single step."""

    if isinstance(zs, np.ndarray) and zs.dtype != object:
        arr = np.asarray(zs, dtype=float)
        if arr.ndim == 1 and dim_z == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[1] != dim_z:
            raise ValueError(
                f"zs must have shape (N, {dim_z})"
                + (" or (N,)" if dim_z == 1 else "")
                + f"; got shape {np.asarray(zs).shape}"
            )
        entries: list[Vector | None] = [np.ascontiguousarray(row) for row in arr]
    else:
        entries = []
        for i, z in enumerate(zs):
            if z is None:
                entries.append(None)
                continue
            row = np.asarray(z, dtype=float).reshape(-1)
            if row.shape[0] != dim_z:
                raise ValueError(
                    f"zs[{i}] must have length {dim_z}; got shape "
                    f"{np.asarray(z).shape}"
                )
            entries.append(row)
    if not entries:
        raise ValueError("zs must contain at least one measurement")

    out: list[Vector | None] = []
    for i, row in enumerate(entries):
        if row is None or np.all(np.isfinite(row)):
            out.append(row)
        elif nan_means_missing and np.all(np.isnan(row)):
            out.append(None)
        else:
            raise ValueError(
                f"zs[{i}] contains non-finite values: {row!r}. Use None "
                "entries (or all-NaN rows with nan_means_missing=True) to "
                "mark missed measurements."
            )
    return out


def _normalize_run_dts(dts: Any, n: int) -> list[float | None]:
    if dts is None:
        return [None] * n
    arr = np.asarray(dts, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("dts must contain only finite values")
    if arr.ndim == 0:
        return [float(arr)] * n
    if arr.shape == (n,):
        return [float(v) for v in arr]
    raise ValueError(f"dts must be a scalar or have shape ({n},); got shape {arr.shape}")


def _normalize_run_covariances(Rs: Any, n: int, dim_z: int) -> list[Matrix | None]:
    if Rs is None:
        return [None] * n
    if isinstance(Rs, np.ndarray) and Rs.ndim == 3:
        if Rs.shape != (n, dim_z, dim_z):
            raise ValueError(
                f"3-D Rs must have shape ({n}, {dim_z}, {dim_z}); got shape {Rs.shape}"
            )
        return [_coerce_square_matrix(m, dim_z, f"Rs[{i}]") for i, m in enumerate(Rs)]
    if isinstance(Rs, (list, tuple)):
        if len(Rs) != n:
            raise ValueError(
                f"list/tuple Rs must have one entry per measurement ({n}); got {len(Rs)}"
            )
        return [_coerce_square_matrix(m, dim_z, f"Rs[{i}]") for i, m in enumerate(Rs)]
    # Scalar or single (dim_z, dim_z) matrix: shared across all steps.
    shared = _coerce_square_matrix(Rs, dim_z, "Rs")
    return [shared] * n


def _run_filter(
    kf: "TurboCKF | TurboSRCKF",
    zs: Any,
    dts: Any,
    Rs: Any,
    fx_args: Sequence[object] | object,
    hx_args: Sequence[object] | object,
    nan_means_missing: bool,
) -> FilterRun:
    entries = _normalize_run_measurements(zs, kf.dim_z, nan_means_missing)
    n = len(entries)
    step_dts = _normalize_run_dts(dts, n)
    step_rs = _normalize_run_covariances(Rs, n, kf.dim_z)

    xs = np.empty((n, kf.dim_x), dtype=float)
    ps = np.empty((n, kf.dim_x, kf.dim_x), dtype=float)
    x_priors = np.empty((n, kf.dim_x), dtype=float)
    p_priors = np.empty((n, kf.dim_x, kf.dim_x), dtype=float)
    lls = np.empty(n, dtype=float)
    nis = np.empty(n, dtype=float)
    missing = np.zeros(n, dtype=bool)

    for i, z in enumerate(entries):
        kf.predict(dt=step_dts[i], fx_args=fx_args)
        kf.update(z, R=step_rs[i], hx_args=hx_args)
        xs[i] = kf.x_post
        ps[i] = kf.P_post
        x_priors[i] = kf.x_prior
        p_priors[i] = kf.P_prior
        lls[i] = kf.log_likelihood
        nis[i] = kf.nis
        missing[i] = z is None

    return FilterRun(
        xs=xs,
        Ps=ps,
        x_priors=x_priors,
        P_priors=p_priors,
        log_likelihoods=lls,
        nis=nis,
        missing=missing,
    )


class _AdaptiveNoiseEstimator:
    """Sage-Husa style adaptive Q/R estimator.

    Maintains an exponentially-weighted moving average of the per-step
    measurement (and optionally process) noise contribution inferred from the
    filter's innovation. After a configurable ``window``-step warm-up the
    estimator writes its current estimate back to the filter's ``R`` (and/or
    ``Q``) covariance on each successful update.

    R-channel uses the residual-based relation of Akhlaghi et al. (2017,
    "Adaptive adjustment of noise covariance in Kalman filter for dynamic
    state estimation"), ``R_est = E[e e^T + H P_post H^T]`` with the
    post-update residual ``e = z - h(x_post)``. For a linear ``h``,
    ``e = R S^-1 y`` and ``H P_post H^T = R - R S^-1 R``, so each step adds
    ``R + R S^-1 (y y^T - S) S^-1 R``, computed from ``y``, ``S`` and the
    ``R`` that built ``S``. That is exact for linear ``h`` and a
    statistical-linearization approximation for a nonlinear ``hx`` (the
    cubature ``S - R`` stands in for ``H P_prior H^T``). Every contribution
    is positive semi-definite because ``S - R`` is, so the estimate cannot
    go indefinite when ``P`` or ``R`` starts overestimated and ``y y^T``
    falls below ``S - R``, which is where the innovation-based
    ``E[y y^T + R - S]`` breaks down.
    For a positive-definite ``R`` the fixed point is the ``R`` at which
    ``E[y y^T] = S``.

    Q-channel uses the state-correction heuristic
    ``Q_est = E[K y y^T K^T]``. This is *not* an unbiased Q estimator and can
    destabilize the filter if used aggressively; callers opting into ``mode in
    ("Q", "both")`` should keep ``alpha`` small and verify NEES on a held-out
    trajectory.

    Each write-back is projected onto the symmetric matrices whose
    eigenvalues are all at least ``diagonal_floor``, so the ``R``/``Q``
    handed to the filter are symmetric positive-definite. The running
    averages behind ``estimate_R()``/``estimate_Q()`` are left unprojected:
    both channels average positive semi-definite contributions already, and
    clipping the running value would carry the added mass forward as bias.
    """

    __slots__ = (
        "window",
        "mode",
        "alpha",
        "diagonal_floor",
        "dim_x",
        "dim_z",
        "_estimate_R",
        "_estimate_Q",
        "_count",
        "_adapt_R",
        "_adapt_Q",
    )

    def __init__(
        self,
        window: int,
        mode: str,
        alpha: float,
        dim_x: int,
        dim_z: int,
        diagonal_floor: float = 1e-12,
    ) -> None:
        window = int(window)
        alpha = float(alpha)
        if window < 1:
            raise ValueError("adaptive window must be at least 1")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if mode not in _ADAPTIVE_MODES:
            raise ValueError(
                f"mode must be one of {_ADAPTIVE_MODES!r}; got {mode!r}"
            )
        if not (diagonal_floor > 0.0 and np.isfinite(diagonal_floor)):
            raise ValueError("diagonal_floor must be finite and positive")

        self.window = window
        self.mode = mode
        self.alpha = alpha
        self.diagonal_floor = float(diagonal_floor)
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self._estimate_R = np.zeros((self.dim_z, self.dim_z), dtype=float)
        self._estimate_Q = np.zeros((self.dim_x, self.dim_x), dtype=float)
        self._count = 0
        self._adapt_R = mode in ("R", "both")
        self._adapt_Q = mode in ("Q", "both")

    @property
    def count(self) -> int:
        return self._count

    @property
    def warmed_up(self) -> bool:
        return self._count >= self.window

    def estimate_R(self) -> Matrix:
        return self._estimate_R.copy()

    def estimate_Q(self) -> Matrix:
        return self._estimate_Q.copy()

    def step(
        self,
        y: Vector,
        S: Matrix,
        R_current: Matrix,
        K: Matrix,
    ) -> tuple[Matrix | None, Matrix | None]:
        """Fold one innovation/update into the running estimate.

        ``R_current`` must be the ``R`` that built ``S`` (the per-call
        override when one was passed to ``update``).

        Returns ``(new_R, new_Q)``. Either entry is ``None`` if (a) that
        channel is disabled, or (b) the estimator is still in the warm-up
        window (``count < window``).
        """

        outer_y = np.outer(y, y)
        if self._adapt_R:
            # e e^T + H P_post H^T written with y, S, R: e = R S^-1 y and
            # H P_post H^T = R - R S^-1 R. S and R are symmetric, so
            # R S^-1 = (S^-1 R)^T.
            try:
                gain = np.linalg.solve(S, R_current).T
            except np.linalg.LinAlgError:
                # Exactly singular S; the backend update used a pseudo-inverse.
                gain = R_current @ np.linalg.pinv(S, hermitian=True)
            contrib_R = R_current + gain @ (outer_y - S) @ gain.T
            self._update_running(self._estimate_R, contrib_R)
        if self._adapt_Q:
            contrib_Q = K @ outer_y @ K.T
            self._update_running(self._estimate_Q, contrib_Q)

        self._count += 1

        new_R: Matrix | None = None
        new_Q: Matrix | None = None
        if self._count >= self.window:
            if self._adapt_R:
                new_R = self._clamp_pd(self._estimate_R)
            if self._adapt_Q:
                new_Q = self._clamp_pd(self._estimate_Q)
        return new_R, new_Q

    def _update_running(self, target: Matrix, contrib: Matrix) -> None:
        # In-place EWMA so callers see the running estimate via estimate_R()/Q().
        contrib_sym = 0.5 * (contrib + contrib.T)
        if self._count == 0:
            np.copyto(target, contrib_sym)
        else:
            target *= 1.0 - self.alpha
            target += self.alpha * contrib_sym

    def _clamp_pd(self, mat: Matrix) -> Matrix:
        # Nearest symmetric matrix (in Frobenius norm) with every eigenvalue
        # >= diagonal_floor. Flooring only the diagonal is not enough: an
        # estimate like [[f, c], [c, f]] with |c| > f stays indefinite.
        sym = 0.5 * (mat + mat.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        out = (eigvecs * np.maximum(eigvals, self.diagonal_floor)) @ eigvecs.T
        return 0.5 * (out + out.T)

    def to_state(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "mode": self.mode,
            "alpha": self.alpha,
            "diagonal_floor": self.diagonal_floor,
            "dim_x": self.dim_x,
            "dim_z": self.dim_z,
            "estimate_R": self._estimate_R.copy(),
            "estimate_Q": self._estimate_Q.copy(),
            "count": self._count,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "_AdaptiveNoiseEstimator":
        est = cls(
            window=int(state["window"]),
            mode=str(state["mode"]),
            alpha=float(state["alpha"]),
            dim_x=int(state["dim_x"]),
            dim_z=int(state["dim_z"]),
            diagonal_floor=float(state.get("diagonal_floor", 1e-12)),
        )
        est._estimate_R = np.array(state["estimate_R"], dtype=float, copy=True)
        est._estimate_Q = np.array(state["estimate_Q"], dtype=float, copy=True)
        est._count = int(state["count"])
        return est


class _ValidatedStateMixin:
    """Coercing property views over the core filter state.

    ``x``, ``P``, ``Q``, ``R`` and ``dt`` accept the spellings people
    actually write (plain lists, integer arrays, column vectors, scalars or
    diagonals for covariances) and normalize them to the float64 layouts the
    Rust backend needs. Invalid shapes, sizes, and non-finite values raise
    at assignment time with the attribute name in the message, rather than
    surfacing later as an opaque conversion error inside predict()/update().
    """

    dim_x: int
    dim_z: int

    @property
    def x(self) -> Vector:
        """State mean, shape ``(dim_x,)``."""

        return self._x

    @x.setter
    def x(self, value: npt.ArrayLike) -> None:
        self._x = _coerce_state_vector(value, self.dim_x, "x")

    @property
    def P(self) -> Matrix:
        """State covariance, shape ``(dim_x, dim_x)``."""

        return self._P

    @P.setter
    def P(self, value: npt.ArrayLike) -> None:
        self._P = _coerce_square_matrix(value, self.dim_x, "P")

    @property
    def Q(self) -> Matrix:
        """Process-noise covariance, shape ``(dim_x, dim_x)``."""

        return self._Q

    @Q.setter
    def Q(self, value: npt.ArrayLike) -> None:
        self._Q = _coerce_square_matrix(value, self.dim_x, "Q")

    @property
    def R(self) -> Matrix:
        """Measurement-noise covariance, shape ``(dim_z, dim_z)``."""

        return self._R

    @R.setter
    def R(self, value: npt.ArrayLike) -> None:
        self._R = _coerce_square_matrix(value, self.dim_z, "R")

    @property
    def dt(self) -> float:
        """Default time step. Assignments are pushed straight to the Rust
        backend, so ``kf.dt = ...`` takes effect for every predict path
        (including the standard/linear-model predicts, which read dt
        backend-side)."""

        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        val = float(value)
        if not np.isfinite(val):
            raise ValueError("dt must be finite")
        self._dt = val
        # The backend does not exist yet during __init__; the constructors
        # pass dt through, so the two stay in sync either way. Pushing here
        # (instead of on every predict/update) keeps the hot loop free of an
        # extra FFI call.
        backend = getattr(self, "_rust_backend", None)
        if backend is not None:
            backend.set_dt(val)


class TurboCKF(_ValidatedStateMixin):
    """Rust-backed Cubature Kalman Filter.

    Callback contract for ``fx`` and ``hx``: both must accept a batch of
    sigma points with shape ``(2 * dim_x, dim_x)`` and return an array of
    shape ``(2 * dim_x, dim_x)`` (``fx``) or ``(2 * dim_x, dim_z)`` (``hx``).
    Pointwise callbacks (one sigma point at a time) are rejected — wrap them
    with ``np.apply_along_axis(..., axis=1)`` or vectorize directly.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dt: float,
        hx: Callable[..., npt.ArrayLike],
        fx: Callable[..., npt.ArrayLike],
    ) -> None:
        if dim_x <= 0 or dim_z <= 0:
            raise ValueError("dim_x and dim_z must be positive")
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")

        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.dt = float(dt)
        self.hx = hx
        self.fx = fx

        self.x: Vector = np.zeros(self.dim_x, dtype=float)
        self.P: Matrix = np.eye(self.dim_x, dtype=float)
        self.Q: Matrix = np.eye(self.dim_x, dtype=float)
        self.R: Matrix = np.eye(self.dim_z, dtype=float)

        self.K: Matrix = np.zeros((self.dim_x, self.dim_z), dtype=float)
        self.y: Vector = np.zeros(self.dim_z, dtype=float)
        self.z: Vector = np.zeros(self.dim_z, dtype=float)
        self.S: Matrix = np.eye(self.dim_z, dtype=float)
        self.SI: Matrix = np.eye(self.dim_z, dtype=float)

        self.x_prior: Vector = self.x.copy()
        self.P_prior: Matrix = self.P.copy()
        self.x_post: Vector = self.x.copy()
        self.P_post: Matrix = self.P.copy()

        self.z_pred: Vector = np.zeros(self.dim_z, dtype=float)
        self.log_likelihood: float = float("nan")
        self.likelihood: float = float("nan")
        self.mahalanobis: float = float("nan")
        self.nis: float = float("nan")

        # stable_cholesky jitter diagnostics — populated from the backend
        # after each predict/update. last_jitter is the jitter applied on the
        # most recent step; max_jitter / jitter_count are cumulative.
        self.last_jitter: float = 0.0
        self.max_jitter: float = 0.0
        self.jitter_count: int = 0
        self.singular_innovation_count: int = 0

        # Adaptive Q/R estimator. None when disabled (the default); set via
        # enable_adaptive_noise().
        self._adaptive: _AdaptiveNoiseEstimator | None = None

        self._rust_backend = _rust.CubatureKalmanFilter(self.dim_x, self.dim_z, self.dt)
        self._backend_name = "rust"

    def __repr__(self) -> str:
        return (
            f"TurboCKF(dim_x={self.dim_x}, dim_z={self.dim_z}, dt={self.dt}, "
            f"backend={self._backend_name!r}, "
            f"log_likelihood={self.log_likelihood:.4g}, jitter_count={self.jitter_count})"
        )

    # ----- prediction ------------------------------------------------------

    def predict(
        self,
        dt: float | None = None,
        fx: Callable[..., npt.ArrayLike] | None = None,
        fx_args: Sequence[object] | object = (),
    ) -> Vector:
        """Run the time-update step."""

        local_dt = self.dt if dt is None else float(dt)
        if not np.isfinite(local_dt):
            raise ValueError("dt must be finite")
        transition = self.fx if fx is None else fx
        args = self._coerce_args(fx_args)

        self._push_state_to_backend()
        self._rust_backend.predict_custom(
            self._make_backend_model(transition, expected_dim=self.dim_x, include_dt=True),
            local_dt,
            args,
        )
        self._pull_state_from_backend()
        return self.x

    def predict_standard_model(self, model_type: str, layout: str = "blocked") -> Vector:
        """Predict with the lightweight (KCKF) linear equations.

        ``model_type`` is ``"constant_velocity"`` or ``"constant_acceleration"``,
        stepped by the current ``dt``. ``layout`` is the order of the state vector:

        - ``"blocked"`` (default): all positions, then all velocities, then all
          accelerations, e.g. ``[x, y, vx, vy]`` or ``[x, y, vx, vy, ax, ay]``.
        - ``"interleaved"``: one block per axis, as in FilterPy, e.g.
          ``[x, vx, y, vy]`` or ``[x, vx, ax, y, vy, ay]``.

        A state stored in the other layout is not detected: the prediction
        silently mixes axes. ``dim_x`` must be a multiple of 2 for constant
        velocity and of 3 for constant acceleration.
        """

        self._validate_standard_model(model_type)
        _validate_standard_layout(layout)
        self._push_state_to_backend()
        self._rust_backend.predict_standard_model(str(model_type), str(layout))
        self._pull_state_from_backend()
        return self.x

    def predict_standard_model_ckf(self, model_type: str, layout: str = "blocked") -> Vector:
        """Predict with the original CKF cubature-point summation equations.

        Takes the same arguments as :meth:`predict_standard_model`. ``layout``
        is ``"blocked"`` (default, ``[x, y, vx, vy]``) or ``"interleaved"``
        (FilterPy order, ``[x, vx, y, vy]``); a state in the other layout
        silently gives wrong predictions.
        """

        self._validate_standard_model(model_type)
        _validate_standard_layout(layout)
        self._push_state_to_backend()
        self._rust_backend.predict_standard_model_ckf(str(model_type), str(layout))
        self._pull_state_from_backend()
        return self.x

    def predict_linear_model(self, f: npt.ArrayLike) -> Vector:
        """Predict using KCKF equations with a caller-provided linear transition matrix."""

        f_mat = self._coerce_covariance(f, self.dim_x, "F")
        self._push_state_to_backend()
        self._rust_backend.predict_linear_model(f_mat)
        self._pull_state_from_backend()
        return self.x

    def predict_linear_model_ckf(self, f: npt.ArrayLike) -> Vector:
        """Predict using CKF cubature summation equations with a caller-provided linear transition matrix."""

        f_mat = self._coerce_covariance(f, self.dim_x, "F")
        self._push_state_to_backend()
        self._rust_backend.predict_linear_model_ckf(f_mat)
        self._pull_state_from_backend()
        return self.x

    # ----- update ----------------------------------------------------------

    def update(
        self,
        z: npt.ArrayLike | None,
        R: npt.ArrayLike | None = None,
        hx: Callable[..., npt.ArrayLike] | None = None,
        hx_args: Sequence[object] | object = (),
    ) -> Vector:
        """Run the measurement-update step.

        Passing ``z=None`` skips the update: ``x_post`` and ``P_post`` are
        snapshotted from the current prior, and all innovation-derived
        diagnostics (``y``, ``S``, ``SI``, ``K``, ``log_likelihood``,
        ``mahalanobis``, ``nis``) are reset to neutral values so they aren't
        silently read as if a real measurement happened.
        """

        if z is None:
            self._push_state_to_backend()
            self._rust_backend.clear_update_diagnostics()
            self._pull_state_from_backend()
            # x_post / P_post mirror the (now unchanged) prior.
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.z = np.full(self.dim_z, np.nan, dtype=float)
            return self.x

        measurement_fn = self.hx if hx is None else hx
        args = self._coerce_args(hx_args)
        z_vec = self._as_vector(z, self.dim_z, "z")
        r_mat = self._coerce_covariance(self.R if R is None else R, self.dim_z, "R")

        self._push_state_to_backend()
        self._rust_backend.update(
            z_vec,
            self._make_backend_model(measurement_fn, expected_dim=self.dim_z, include_dt=False),
            r_mat,
            args,
        )
        self._pull_state_from_backend()
        self._apply_adaptive_noise(r_mat)
        return self.x

    def update_paper_ahrs(
        self, z: npt.ArrayLike, sigma_acc2: float, sigma_mag2: float
    ) -> Vector:
        """Run Eq. (9), (12)-(14) AHRS update in Rust.

        Args:
            z: ``[ax, ay, az, mx, my, mz]``, accelerometer then magnetometer,
                in any units (for example m/s^2 and uT). Each 3-vector is
                normalized to unit length before the update, because the
                observation model predicts unit vectors. The recorded
                ``self.z`` and ``self.y`` use the normalized values. A zero or
                non-finite norm raises ``ValueError``.
            sigma_acc2: variance of each unit-vector accelerometer component.
            sigma_mag2: variance of each unit-vector magnetometer component.

        ``R`` is overwritten with a diagonal matrix holding ``sigma_acc2`` in
        the first three entries and ``sigma_mag2`` in the last three.
        """

        if self.dim_x != 4 or self.dim_z != 6:
            raise ValueError("update_paper_ahrs requires dim_x == 4 and dim_z == 6")
        sigma_acc2 = float(sigma_acc2)
        sigma_mag2 = float(sigma_mag2)
        if not np.isfinite(sigma_acc2) or sigma_acc2 <= 0.0:
            raise ValueError("sigma_acc2 must be finite and positive")
        if not np.isfinite(sigma_mag2) or sigma_mag2 <= 0.0:
            raise ValueError("sigma_mag2 must be finite and positive")
        z_vec = self._as_vector(z, self.dim_z, "z")
        self._push_state_to_backend()
        self._rust_backend.update_paper_ahrs(z_vec, sigma_acc2, sigma_mag2)
        self._pull_state_from_backend()
        return self.x

    def run(
        self,
        zs: npt.ArrayLike | Sequence[npt.ArrayLike | None],
        dts: npt.ArrayLike | None = None,
        Rs: npt.ArrayLike | None = None,
        fx_args: Sequence[object] | object = (),
        hx_args: Sequence[object] | object = (),
        nan_means_missing: bool = False,
    ) -> FilterRun:
        """Run predict+update over a whole measurement sequence in one call.

        Equivalent to the hand-written loop, with history collection and
        missed-measurement handling built in::

            result = kf.run(zs)
            result.xs       # (N, dim_x) posterior means
            result.Ps       # (N, dim_x, dim_x) posterior covariances

        The filter instance is mutated step by step exactly as if
        :meth:`predict` and :meth:`update` had been called in a loop, so the
        final state is available on ``self`` afterwards, and features like
        adaptive noise keep working. All input validation happens up front —
        a shape error raises before the filter has advanced at all.

        For *linear* models prefer the static :meth:`batch_filter`, which
        runs the whole loop inside Rust in a single crossing.

        Args:
            zs: measurement sequence. Either an ``(N, dim_z)`` array
                (``(N,)`` is also accepted when ``dim_z == 1``), or a list
                whose entries are length-``dim_z`` vectors (scalars when
                ``dim_z == 1``) or ``None`` for a missed measurement. A
                ``None`` entry runs the predict step and skips the update.
            dts: optional per-step time steps — a scalar or an ``(N,)``
                array. Defaults to ``self.dt`` for every step.
            Rs: optional measurement noise — a scalar or ``(dim_z, dim_z)``
                matrix shared by all steps, or an ``(N, dim_z, dim_z)``
                array / length-``N`` list for per-step values. Defaults to
                ``self.R``.
            fx_args: extra positional args forwarded to ``fx`` each step.
            hx_args: extra positional args forwarded to ``hx`` each step.
            nan_means_missing: when True, rows of ``zs`` that are entirely
                NaN are treated as missed measurements instead of raising.
                Rows with a *mix* of NaN and finite values always raise.

        Returns:
            :class:`FilterRun` with stacked ``xs``, ``Ps``, ``x_priors``,
            ``P_priors``, ``log_likelihoods``, ``nis``, and the boolean
            ``missing`` mask. Likelihood/NIS entries are NaN at missed
            steps.
        """

        return _run_filter(self, zs, dts, Rs, fx_args, hx_args, nan_means_missing)

    # ----- diagnostics / utility ------------------------------------------

    def enable_adaptive_noise(
        self,
        window: int = 30,
        mode: str = "R",
        alpha: float = 0.3,
        diagonal_floor: float = 1e-12,
    ) -> None:
        """Turn on Sage-Husa adaptive R (and/or Q) estimation.

        After each successful ``update(z=...)`` call, folds the new
        innovation into a running EWMA estimate of the measurement-noise
        covariance ``R`` (and/or process-noise covariance ``Q``). For the
        first ``window`` updates, only the estimator state is built up — the
        filter's ``R``/``Q`` are left untouched. From step ``window`` onward,
        the current estimate is written back to ``self.R`` (and/or
        ``self.Q``) before the next predict/update.

        Off by default. Existing per-step behaviour and all existing tests
        are unaffected when this is not called.

        Args:
            window: warm-up step count. The estimator accumulates this many
                innovations before writing the first update back to
                ``R``/``Q``. Setting this larger trades adaptation latency
                for less noise in the initial estimate. Default 30.
            mode: which noise covariance to adapt. One of ``"R"``
                (residual-based, after Akhlaghi et al. 2017: each update
                adds ``e e^T + H P_post H^T``, which is positive
                semi-definite, so the estimate cannot go indefinite when
                ``R`` or ``P`` starts overestimated; exact for a linear
                ``hx``, a statistical-linearization approximation otherwise),
                ``"Q"`` (heuristic — state-correction outer-product; can
                destabilize, keep ``alpha`` small and verify NEES on a
                held-out trajectory), or ``"both"``. Default ``"R"``.
            alpha: EWMA forgetting factor in ``(0, 1]``. Larger values track
                changes faster but produce noisier estimates. Typical
                ``0.01..0.3``. Default ``0.3``.
            diagonal_floor: eigenvalue floor for every write-back. The
                estimate is projected onto the symmetric matrices whose
                eigenvalues are all at least this value, so the ``R``/``Q``
                written to the filter are symmetric positive-definite.
                Despite the name, it bounds eigenvalues, not only diagonal
                entries. Default ``1e-12``.

        Notes:
            - Adaptation uses the ``R`` actually applied on each update.
              When ``update(z, R=...)`` or ``run(zs, Rs=...)`` overrides
              ``R`` for a step, the estimate is formed against that
              override; the written-back estimate still goes to ``self.R``.
            - Adaptive estimation runs on the standard ``update(z=...)``
              path only. ``update_paper_ahrs`` overwrites ``R`` from its
              ``sigma_acc2``/``sigma_mag2`` arguments on every call, so the
              adaptive R-write would be discarded; the estimator is
              skipped there.
            - The ``batch_filter`` / ``batch_parallel_step`` paths use
              their own Rust-side ``R``; adaptive noise is per-instance
              and does not affect those static-method calls.
        """

        self._adaptive = _AdaptiveNoiseEstimator(
            window=window,
            mode=mode,
            alpha=alpha,
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            diagonal_floor=diagonal_floor,
        )

    def disable_adaptive_noise(self) -> None:
        """Turn adaptive R/Q estimation off and drop the estimator state."""

        self._adaptive = None

    @property
    def adaptive_noise_estimator(self) -> _AdaptiveNoiseEstimator | None:
        """Read-only handle to the current adaptive estimator (or ``None``)."""

        return self._adaptive

    def _apply_adaptive_noise(self, R_used: Matrix | None = None) -> None:
        # R_used is the R that built this update's S: the per-call override
        # when one was passed, or None when the stored self.R was used.
        # Passing self.R under an override skews the estimate by the gap.
        if self._adaptive is None:
            return
        if not (np.all(np.isfinite(self.y)) and np.all(np.isfinite(self.S))):
            return
        new_R, new_Q = self._adaptive.step(
            y=self.y,
            S=self.S,
            R_current=self.R if R_used is None else R_used,
            K=self.K,
        )
        if new_R is not None:
            self.R = new_R
        if new_Q is not None:
            self.Q = new_Q

    def gate(self, threshold: float) -> bool:
        """Chi-square gating decision on the most recent innovation.

        Returns ``True`` if the squared Mahalanobis distance (NIS) is below
        the supplied chi-square threshold for ``dim_z`` degrees of freedom.
        Returns ``False`` if no update has happened yet, if NIS is NaN, or
        if it exceeds the threshold.
        """

        if not np.isfinite(self.nis):
            return False
        return float(self.nis) <= float(threshold)

    def reset(self, x: npt.ArrayLike | None = None, P: npt.ArrayLike | None = None) -> None:
        """Reset state to the constructor defaults (or supplied values) and
        clear all cached diagnostics. ``Q``, ``R``, ``dt``, ``fx``, ``hx``
        are preserved."""

        self.x = (
            np.zeros(self.dim_x, dtype=float)
            if x is None
            else self._as_vector(x, self.dim_x, "x")
        )
        self.P = (
            np.eye(self.dim_x, dtype=float)
            if P is None
            else self._coerce_covariance(P, self.dim_x, "P")
        )
        self.K = np.zeros((self.dim_x, self.dim_z), dtype=float)
        self.y = np.zeros(self.dim_z, dtype=float)
        self.z = np.zeros(self.dim_z, dtype=float)
        self.S = np.eye(self.dim_z, dtype=float)
        self.SI = np.eye(self.dim_z, dtype=float)
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self.z_pred = np.zeros(self.dim_z, dtype=float)
        self.log_likelihood = float("nan")
        self.likelihood = float("nan")
        self.mahalanobis = float("nan")
        self.nis = float("nan")
        self.last_jitter = 0.0
        self.max_jitter = 0.0
        self.jitter_count = 0
        self.singular_innovation_count = 0
        # Drop adaptive estimator state on reset — Q/R are back to defaults so
        # an existing accumulator would carry stale evidence into a fresh run.
        # Estimator config (window/mode/alpha) is forgotten too; users that
        # want it back should re-call enable_adaptive_noise() after reset().
        self._adaptive = None
        # Rebuild the backend to drop its accumulated counters too.
        self._rust_backend = _rust.CubatureKalmanFilter(self.dim_x, self.dim_z, self.dt)
        self._push_state_to_backend()

    def copy(self) -> "TurboCKF":
        """Return an independent filter with the same state, dimensions,
        callbacks, and diagnostics. Useful for Monte-Carlo runs."""

        new = TurboCKF(self.dim_x, self.dim_z, self.dt, hx=self.hx, fx=self.fx)
        new.x = self.x.copy()
        new.P = self.P.copy()
        new.Q = self.Q.copy()
        new.R = self.R.copy()
        new.K = self.K.copy()
        new.y = self.y.copy()
        new.z = self.z.copy()
        new.S = self.S.copy()
        new.SI = self.SI.copy()
        new.x_prior = self.x_prior.copy()
        new.P_prior = self.P_prior.copy()
        new.x_post = self.x_post.copy()
        new.P_post = self.P_post.copy()
        new.z_pred = self.z_pred.copy()
        new.log_likelihood = self.log_likelihood
        new.likelihood = self.likelihood
        new.mahalanobis = self.mahalanobis
        new.nis = self.nis
        new.last_jitter = self.last_jitter
        new.max_jitter = self.max_jitter
        new.jitter_count = self.jitter_count
        new.singular_innovation_count = self.singular_innovation_count
        if self._adaptive is not None:
            new._adaptive = _AdaptiveNoiseEstimator.from_state(self._adaptive.to_state())
        new._push_state_to_backend()
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> "TurboCKF":
        return self.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the filter state to a plain dict (ndarrays kept as
        ndarrays). Callbacks are *not* included — restoring requires the
        caller to re-supply them via :meth:`from_dict`."""

        out = {
            "version": 1,
            "dim_x": self.dim_x,
            "dim_z": self.dim_z,
            "dt": self.dt,
            "x": self.x.copy(),
            "P": self.P.copy(),
            "Q": self.Q.copy(),
            "R": self.R.copy(),
            "x_prior": self.x_prior.copy(),
            "P_prior": self.P_prior.copy(),
            "x_post": self.x_post.copy(),
            "P_post": self.P_post.copy(),
            "log_likelihood": self.log_likelihood,
            "likelihood": self.likelihood,
            "mahalanobis": self.mahalanobis,
            "nis": self.nis,
            "jitter_count": self.jitter_count,
            "max_jitter": self.max_jitter,
            "singular_innovation_count": self.singular_innovation_count,
        }
        if self._adaptive is not None:
            out["adaptive"] = self._adaptive.to_state()
        return out

    @classmethod
    def from_dict(
        cls,
        state: Mapping[str, Any],
        hx: Callable[..., npt.ArrayLike],
        fx: Callable[..., npt.ArrayLike],
    ) -> "TurboCKF":
        """Reconstruct a filter from :meth:`to_dict` output."""

        version = state.get("version", 1)
        if version != 1:
            raise ValueError(f"unsupported TurboCKF dict version: {version!r}")
        kf = cls(
            dim_x=int(state["dim_x"]),
            dim_z=int(state["dim_z"]),
            dt=float(state["dt"]),
            hx=hx,
            fx=fx,
        )
        kf.x = np.array(state["x"], dtype=float, copy=True).reshape(-1)
        kf.P = np.array(state["P"], dtype=float, copy=True)
        kf.Q = np.array(state["Q"], dtype=float, copy=True)
        kf.R = np.array(state["R"], dtype=float, copy=True)
        kf.x_prior = np.array(state.get("x_prior", kf.x), dtype=float, copy=True).reshape(-1)
        kf.P_prior = np.array(state.get("P_prior", kf.P), dtype=float, copy=True)
        kf.x_post = np.array(state.get("x_post", kf.x), dtype=float, copy=True).reshape(-1)
        kf.P_post = np.array(state.get("P_post", kf.P), dtype=float, copy=True)
        kf.log_likelihood = float(state.get("log_likelihood", float("nan")))
        kf.likelihood = float(state.get("likelihood", float("nan")))
        kf.mahalanobis = float(state.get("mahalanobis", float("nan")))
        kf.nis = float(state.get("nis", float("nan")))
        kf.jitter_count = int(state.get("jitter_count", 0))
        kf.max_jitter = float(state.get("max_jitter", 0.0))
        kf.singular_innovation_count = int(state.get("singular_innovation_count", 0))
        adaptive_state = state.get("adaptive")
        if adaptive_state is not None:
            kf._adaptive = _AdaptiveNoiseEstimator.from_state(adaptive_state)
        kf._push_state_to_backend()
        return kf

    @staticmethod
    def batch_filter(
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        zs: npt.ArrayLike,
        F: npt.ArrayLike,
        H: npt.ArrayLike,
        Q: npt.ArrayLike | None = None,
        R: npt.ArrayLike | None = None,
    ) -> tuple[Matrix, np.ndarray, Vector]:
        """Linear Kalman batch filter — one Rust-side pass over ``zs``.

        Runs the full predict/update loop inside the backend so per-step
        Python ↔ Rust crossings disappear. For the standard linear case
        this is the order-of-magnitude path the audit called out.

        ``F``, ``H``, ``Q``, ``R`` may be either constant matrices or
        per-step arrays with leading dimension ``N``. Each step predicts
        then updates, so ``F[k]`` and ``Q[k]`` are the step ``k-1 -> k``
        transition (``F[0]`` maps ``x0`` to step 0). This is the FilterPy
        convention, and ``rts_smooth`` accepts the same length-``N`` arrays::

            xs, Ps, _ = TurboCKF.batch_filter(x0, P0, zs, Fs, H, Qs, R)
            xs_s, Ps_s = TurboCKF.rts_smooth(xs, Ps, Fs, Qs)

        For nonlinear ``fx`` / ``hx``, use the per-step ``predict()`` /
        ``update()`` API on a :class:`TurboCKF` instance.

        Args:
            x0: initial state, shape ``(dim_x,)``.
            P0: initial covariance, shape ``(dim_x, dim_x)``.
            zs: observations, shape ``(N, dim_z)``.
            F: transition, shape ``(dim_x, dim_x)`` or
                ``(N, dim_x, dim_x)``.
            H: measurement, shape ``(dim_z, dim_x)`` or
                ``(N, dim_z, dim_x)``.
            Q: process noise, shape ``(dim_x, dim_x)`` or
                ``(N, dim_x, dim_x)``. Defaults to zeros.
            R: measurement noise, shape ``(dim_z, dim_z)`` or
                ``(N, dim_z, dim_z)``. Defaults to identity.

        Returns:
            ``(xs, Ps, log_likelihoods)`` of shapes ``(N, dim_x)``,
            ``(N, dim_x, dim_x)``, ``(N,)``.
        """

        x0_arr = np.ascontiguousarray(np.asarray(x0, dtype=float)).reshape(-1)
        p0_arr = np.ascontiguousarray(np.asarray(P0, dtype=float))
        zs_arr = np.ascontiguousarray(np.asarray(zs, dtype=float))
        if zs_arr.ndim != 2:
            raise ValueError(
                f"zs must be 2D with shape (N, dim_z); got ndim={zs_arr.ndim}"
            )
        n, dim_z = zs_arr.shape
        dim_x = x0_arr.shape[0]
        if dim_x == 0:
            raise ValueError("x0 must have at least one element")
        if n == 0:
            raise ValueError("zs must contain at least one observation")
        if p0_arr.shape != (dim_x, dim_x):
            raise ValueError(
                f"P0 must have shape ({dim_x}, {dim_x}); got {p0_arr.shape}"
            )

        def _broadcast(name: str, value: npt.ArrayLike, inner: tuple[int, ...]) -> np.ndarray:
            arr = np.ascontiguousarray(np.asarray(value, dtype=float))
            if arr.ndim == 2:
                if arr.shape != inner:
                    raise ValueError(
                        f"{name} must have shape {inner} or ({n}, *{inner}); got {arr.shape}"
                    )
                return np.ascontiguousarray(np.broadcast_to(arr, (n,) + inner))
            if arr.ndim == 3:
                if arr.shape != (n,) + inner:
                    raise ValueError(
                        f"{name} must have shape {inner} or ({n}, *{inner}); got {arr.shape}"
                    )
                return arr
            raise ValueError(
                f"{name} must be 2D or 3D; got ndim={arr.ndim}"
            )

        fs = _broadcast("F", F, (dim_x, dim_x))
        hs = _broadcast("H", H, (dim_z, dim_x))
        qs = _broadcast(
            "Q", Q if Q is not None else np.zeros((dim_x, dim_x)), (dim_x, dim_x)
        )
        rs = _broadcast(
            "R", R if R is not None else np.eye(dim_z), (dim_z, dim_z)
        )

        xs, Ps, lls = _rust.batch_filter_linear(x0_arr, p0_arr, zs_arr, fs, hs, qs, rs)
        return (
            np.asarray(xs, dtype=float),
            np.asarray(Ps, dtype=float),
            np.asarray(lls, dtype=float),
        )

    @staticmethod
    def batch_parallel_step(
        xs: npt.ArrayLike,
        Ps: npt.ArrayLike,
        zs: npt.ArrayLike,
        F: npt.ArrayLike,
        H: npt.ArrayLike,
        Q: npt.ArrayLike | None = None,
        R: npt.ArrayLike | None = None,
    ) -> tuple[Matrix, np.ndarray, Vector, np.ndarray]:
        """Parallel linear predict+update across a bank of M independent KFs.

        The "many filters, one observation each" pattern (Monte-Carlo banks,
        particle filters, multi-target tracking). Distinct from
        :meth:`batch_filter` ("one filter, many observations") — here every
        filter advances by exactly one predict + linear update against its
        own ``z_i``, and the bank shares a single ``(F, H, Q, R)``.
        The M steps run in parallel via rayon with the GIL released.

        Args:
            xs: prior states, shape ``(M, dim_x)``.
            Ps: prior covariances, shape ``(M, dim_x, dim_x)``.
            zs: per-filter observations, shape ``(M, dim_z)``.
            F: shared transition, shape ``(dim_x, dim_x)``.
            H: shared measurement, shape ``(dim_z, dim_x)``.
            Q: shared process noise, shape ``(dim_x, dim_x)``. Defaults to
                zeros.
            R: shared measurement noise, shape ``(dim_z, dim_z)``. Defaults
                to identity.

        Returns:
            ``(xs_new, Ps_new, log_likelihoods, status)`` with shapes
            ``(M, dim_x)``, ``(M, dim_x, dim_x)``, ``(M,)``, ``(M,)``.

            ``status[i]`` reports per-filter health:

            * ``0`` — innovation covariance was PD (Cholesky succeeded).
            * ``1`` — innovation covariance was singular; used the
              pseudo-inverse fallback for ``K``. Treat as a soft warning.
            * ``2`` — no inverse at all; the measurement update was
              **skipped** and ``log_likelihoods[i] = -inf``. The returned
              ``(xs_new[i], Ps_new[i])`` is the predict-step output only.

            One bad filter does not abort the bank — Monte-Carlo callers
            can mask on ``status != 2`` and keep going.
        """

        xs_arr = np.ascontiguousarray(np.asarray(xs, dtype=float))
        ps_arr = np.ascontiguousarray(np.asarray(Ps, dtype=float))
        zs_arr = np.ascontiguousarray(np.asarray(zs, dtype=float))
        f_arr = np.ascontiguousarray(np.asarray(F, dtype=float))
        h_arr = np.ascontiguousarray(np.asarray(H, dtype=float))

        if xs_arr.ndim != 2:
            raise ValueError(
                f"xs must be 2D with shape (M, dim_x); got ndim={xs_arr.ndim}"
            )
        m, dim_x = xs_arr.shape
        if m == 0:
            raise ValueError("xs must contain at least one filter")
        if dim_x == 0:
            raise ValueError("dim_x must be positive")

        if zs_arr.ndim != 2 or zs_arr.shape[0] != m:
            raise ValueError(
                f"zs must have shape ({m}, dim_z); got {zs_arr.shape}"
            )
        dim_z = zs_arr.shape[1]
        if dim_z == 0:
            raise ValueError("dim_z must be positive")

        if ps_arr.shape != (m, dim_x, dim_x):
            raise ValueError(
                f"Ps must have shape ({m}, {dim_x}, {dim_x}); got {ps_arr.shape}"
            )
        if f_arr.shape != (dim_x, dim_x):
            raise ValueError(
                f"F must have shape ({dim_x}, {dim_x}); got {f_arr.shape}"
            )
        if h_arr.shape != (dim_z, dim_x):
            raise ValueError(
                f"H must have shape ({dim_z}, {dim_x}); got {h_arr.shape}"
            )

        q_arr = np.ascontiguousarray(
            np.asarray(Q if Q is not None else np.zeros((dim_x, dim_x)), dtype=float)
        )
        r_arr = np.ascontiguousarray(
            np.asarray(R if R is not None else np.eye(dim_z), dtype=float)
        )
        if q_arr.shape != (dim_x, dim_x):
            raise ValueError(
                f"Q must have shape ({dim_x}, {dim_x}); got {q_arr.shape}"
            )
        if r_arr.shape != (dim_z, dim_z):
            raise ValueError(
                f"R must have shape ({dim_z}, {dim_z}); got {r_arr.shape}"
            )

        xs_new, ps_new, lls, status = _rust.batch_parallel_step(
            xs_arr, ps_arr, zs_arr, f_arr, h_arr, q_arr, r_arr
        )
        return (
            np.asarray(xs_new, dtype=float),
            np.asarray(ps_new, dtype=float),
            np.asarray(lls, dtype=float),
            np.asarray(status, dtype=np.int64),
        )

    @staticmethod
    def rts_smooth(
        xs: npt.ArrayLike,
        Ps: npt.ArrayLike,
        Fs: npt.ArrayLike,
        Qs: npt.ArrayLike,
    ) -> tuple[Matrix, np.ndarray]:
        """Rauch-Tung-Striebel fixed-interval smoother.

        Backward pass over a forward-filtered trace. Per-step ``Fs``/``Qs``
        let the smoother handle non-constant transitions (use ``np.tile`` if
        F and Q are actually constant).

        Args:
            xs: filtered state means, shape ``(N, dim_x)``.
            Ps: filtered covariances, shape ``(N, dim_x, dim_x)``.
            Fs: per-step transition matrices, in one of two layouts:

                - ``(N, dim_x, dim_x)``: ``Fs[k]`` maps step ``k-1`` to ``k``,
                  so the ``k -> k+1`` step uses ``Fs[k+1]`` and ``Fs[0]`` is
                  unused. This is the layout ``batch_filter`` consumes and
                  FilterPy's ``rts_smoother`` expects, so the same array can
                  be passed to both.
                - ``(N-1, dim_x, dim_x)``: ``Fs[k]`` maps step ``k`` to
                  ``k+1``. A length-``N`` array ``Fs`` is equivalent to
                  ``Fs[1:]`` here.
            Qs: per-step process-noise covariances. Same layouts as ``Fs``,
                with ``Qs[k]`` added in the step that ``Fs[k]`` describes.
                The layout of ``Qs`` is taken from its own length, not from
                ``Fs``.

        Returns:
            ``(xs_smooth, Ps_smooth)`` with the same shapes as ``(xs, Ps)``.
        """

        xs_arr = np.ascontiguousarray(np.asarray(xs, dtype=float))
        ps_arr = np.ascontiguousarray(np.asarray(Ps, dtype=float))
        fs_arr = np.ascontiguousarray(np.asarray(Fs, dtype=float))
        qs_arr = np.ascontiguousarray(np.asarray(Qs, dtype=float))

        if xs_arr.ndim != 2:
            raise ValueError(
                f"xs must be 2D with shape (N, dim_x); got ndim={xs_arr.ndim}"
            )
        n, dim_x = xs_arr.shape
        if n == 0:
            raise ValueError("xs must contain at least one filtered state")

        if ps_arr.shape != (n, dim_x, dim_x):
            raise ValueError(
                f"Ps must have shape ({n}, {dim_x}, {dim_x}); got {ps_arr.shape}"
            )
        for name, arr in (("Fs", fs_arr), ("Qs", qs_arr)):
            if arr.ndim != 3 or arr.shape[1:] != (dim_x, dim_x) or arr.shape[0] not in (n, max(n - 1, 0)):
                raise ValueError(
                    f"{name} must have shape (N, {dim_x}, {dim_x}) or "
                    f"(N-1, {dim_x}, {dim_x}); got {arr.shape}"
                )

        xs_smooth, ps_smooth = _rust.rts_smooth(xs_arr, ps_arr, fs_arr, qs_arr)
        return np.asarray(xs_smooth, dtype=float), np.asarray(ps_smooth, dtype=float)

    def normalize_state_quaternion(self) -> Vector:
        """Normalize the state when it represents a quaternion."""

        if self.dim_x != 4:
            raise ValueError("normalize_state_quaternion requires dim_x == 4")
        norm = float(np.linalg.norm(self.x))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("quaternion norm must be finite and positive")
        self.x = self.x / norm
        # Keep x_post in sync so introspection reads aren't stale until the
        # next snapshot from the backend.
        self.x_post = self.x.copy()
        return self.x

    def normalize_state_quaternion_backend(self) -> Vector:
        """Normalize quaternion state directly inside the Rust backend."""

        if self.dim_x != 4:
            raise ValueError("normalize_state_quaternion_backend requires dim_x == 4")
        self._push_state_to_backend()
        self._rust_backend.normalize_quaternion_state()
        self._pull_state_from_backend()
        return self.x

    # ----- internals -------------------------------------------------------

    def _push_state_to_backend(self) -> None:
        self._rust_backend.set_state(self._x, self._P, self._Q, self._R)

    def _pull_state_from_backend(self) -> None:
        # The Rust snapshot already returns fresh numpy buffers (via
        # `ToPyArray::to_pyarray`, which allocates a new PyArray per call),
        # so `np.asarray` here is a zero-copy adoption — no aliasing risk to
        # the backend struct's internal storage. Writes go to the private
        # slots directly: backend output is already float64 and well-shaped,
        # so the coercing property setters would only add per-step overhead.
        snap = self._rust_backend.snapshot()
        self._x = np.asarray(snap["x"], dtype=float).reshape(-1)
        self._P = np.asarray(snap["P"], dtype=float)
        self._Q = np.asarray(snap["Q"], dtype=float)
        self._R = np.asarray(snap["R"], dtype=float)
        self.K = np.asarray(snap["K"], dtype=float)
        self.y = np.asarray(snap["y"], dtype=float).reshape(-1)
        self.z = np.asarray(snap["z"], dtype=float).reshape(-1)
        self.S = np.asarray(snap["S"], dtype=float)
        self.SI = np.asarray(snap["SI"], dtype=float)
        self.x_prior = np.asarray(snap["x_prior"], dtype=float).reshape(-1)
        self.P_prior = np.asarray(snap["P_prior"], dtype=float)
        self.x_post = np.asarray(snap["x_post"], dtype=float).reshape(-1)
        self.P_post = np.asarray(snap["P_post"], dtype=float)
        self.z_pred = np.asarray(snap["z_pred"], dtype=float).reshape(-1)
        self.log_likelihood = float(snap["log_likelihood"])
        self.likelihood = float(snap["likelihood"])
        self.mahalanobis = float(snap["mahalanobis"])
        self.nis = float(snap["nis"])
        self.last_jitter = float(snap["last_jitter"])
        self.max_jitter = float(snap["max_jitter"])
        self.jitter_count = int(snap["jitter_count"])
        self.singular_innovation_count = int(snap["singular_innovation_count"])

    def _make_backend_model(
        self,
        model: Callable[..., npt.ArrayLike],
        expected_dim: int,
        include_dt: bool,
    ) -> Callable[..., Matrix]:
        def _wrapped(sigma_points, *call_args):
            sigma = np.asarray(sigma_points, dtype=float)
            if sigma.ndim != 2:
                raise ValueError(
                    "fx/hx must accept a 2D batch of sigma points with shape "
                    f"(2 * dim_x, dim_x); got ndim={sigma.ndim}. See README "
                    "for the vectorized-callback contract."
                )
            if include_dt:
                if len(call_args) == 0:
                    raise ValueError("missing dt argument for transition callback")
                local_dt = float(call_args[0])
                extra_args = call_args[1:]
            else:
                local_dt = 0.0
                extra_args = call_args
            return self._apply_model(
                model=model,
                sigma_points=sigma,
                expected_dim=expected_dim,
                include_dt=include_dt,
                dt=local_dt,
                extra_args=extra_args,
            )

        return _wrapped

    def _apply_model(
        self,
        model: Callable[..., npt.ArrayLike],
        sigma_points: Matrix,
        expected_dim: int,
        include_dt: bool,
        dt: float,
        extra_args: Sequence[object],
    ) -> Matrix:
        args = tuple(extra_args or ())
        expected_shape = (sigma_points.shape[0], expected_dim)

        if include_dt:
            raw = model(sigma_points, dt, *args)
        else:
            raw = model(sigma_points, *args)

        arr = np.asarray(raw, dtype=float)
        if arr.shape != expected_shape:
            raise ValueError(
                f"fx/hx must return shape {expected_shape}, got {arr.shape}. "
                "TurboCKF requires vectorized callbacks — write fx/hx so that "
                "they map a batch of sigma points (rows) to a batch of outputs "
                "(rows). Pointwise callbacks are not supported; wrap with "
                "np.apply_along_axis or vectorize directly."
            )
        return arr

    # ----- introspection helpers (kept for parity / test access) ----------

    def _cubature_points(self, mean: npt.ArrayLike, cov: npt.ArrayLike) -> Matrix:
        """Compute cubature sigma points for arbitrary (mean, cov). Provided
        so users and tests can reproduce / inspect the sigma set without
        going through the full predict step. The actual filter math runs in
        Rust."""

        x = self._as_vector(mean, self.dim_x, "mean")
        p = self._coerce_covariance(cov, self.dim_x, "cov")
        chol = self._stable_cholesky(p)
        scale = np.sqrt(float(self.dim_x))
        offsets = scale * chol.T
        plus = x + offsets
        minus = x - offsets
        return np.vstack([plus, minus]).astype(float, copy=False)

    def _apply_transition(
        self,
        fx: Callable[..., npt.ArrayLike],
        sigma_points: Matrix,
        dt: float,
        fx_args: Sequence[object],
    ) -> Matrix:
        return self._apply_model(
            model=fx,
            sigma_points=sigma_points,
            expected_dim=self.dim_x,
            include_dt=True,
            dt=dt,
            extra_args=fx_args,
        )

    def _apply_measurement(
        self,
        hx: Callable[..., npt.ArrayLike],
        sigma_points: Matrix,
        hx_args: Sequence[object],
    ) -> Matrix:
        return self._apply_model(
            model=hx,
            sigma_points=sigma_points,
            expected_dim=self.dim_z,
            include_dt=False,
            dt=0.0,
            extra_args=hx_args,
        )

    @staticmethod
    def _stable_cholesky(cov: Matrix) -> Matrix:
        jitter = 0.0
        eye = np.eye(cov.shape[0], dtype=float)
        for _ in range(6):
            try:
                return np.linalg.cholesky(cov + jitter * eye)
            except np.linalg.LinAlgError:
                jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
        return np.linalg.cholesky(cov + 1e-6 * eye)

    @staticmethod
    def _validate_standard_model(model_type: str) -> None:
        name = str(model_type)
        if name not in _STANDARD_MODELS:
            choices = ", ".join(repr(m) for m in _STANDARD_MODELS)
            raise ValueError(
                f"unsupported model_type {name!r}; expected one of {choices}"
            )

    @staticmethod
    def _as_vector(value: npt.ArrayLike, size: int, name: str) -> Vector:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape[0] != size:
            raise ValueError(f"{name} must have length {size}, got shape {arr.shape}")
        return arr

    @staticmethod
    def _coerce_covariance(value: npt.ArrayLike, size: int, name: str) -> Matrix:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr) * np.eye(size, dtype=float)
        if arr.shape != (size, size):
            raise ValueError(f"{name} must have shape {(size, size)}, got {arr.shape}")
        return arr

    @staticmethod
    def _coerce_args(args: Sequence[object] | object | None) -> tuple[object, ...]:
        """Coerce a callback-args specifier into a tuple of positional args.

        - ``None`` or empty tuple/list → ``()``
        - tuple/list → unpacked into positional args (matches FilterPy)
        - numpy ndarray → single positional arg (a list-of-floats would be
          ambiguous; ndarrays are almost always the whole arg the user means)
        - anything else → single positional arg
        """

        if args is None:
            return ()
        if isinstance(args, tuple):
            return args
        if isinstance(args, list):
            return tuple(args)
        return (args,)


class TurboSRCKF(_ValidatedStateMixin):
    """Square-root Cubature Kalman Filter (SR-CKF).

    Propagates the lower-triangular Cholesky factor of P directly instead of
    P itself. Predict step uses a single QR of stacked weighted sigma-point
    deltas + ``chol(Q)``; update uses one QR for the innovation factor plus
    ``dim_z`` rank-1 Cholesky downdates for the posterior factor. The full
    filter loop never calls ``stable_cholesky`` on P — so the silent
    jitter-on-the-diagonal hazard that the standard :class:`TurboCKF`
    accumulates at every predict simply doesn't exist here.

    Same vectorised callback contract as :class:`TurboCKF`: ``fx`` and ``hx``
    take ``(2 * dim_x, dim_x)`` batches of sigma points and return
    ``(2 * dim_x, dim_x)`` and ``(2 * dim_x, dim_z)`` respectively.

    Only the ``predict_custom`` + ``update`` API surface from TurboCKF is
    mirrored here. For linear closed-form predicts or the paper AHRS update
    use :class:`TurboCKF` (the silent-jitter blast radius on those paths is
    bounded by per-step measurements anyway, so the SR variant is lower
    leverage).
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dt: float,
        hx: Callable[..., npt.ArrayLike],
        fx: Callable[..., npt.ArrayLike],
    ) -> None:
        if dim_x <= 0 or dim_z <= 0:
            raise ValueError("dim_x and dim_z must be positive")
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")

        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.dt = float(dt)
        self.hx = hx
        self.fx = fx

        # State + factor view.
        self.x: Vector = np.zeros(self.dim_x, dtype=float)
        self.P: Matrix = np.eye(self.dim_x, dtype=float)
        self.chol_P: Matrix = np.eye(self.dim_x, dtype=float)
        self.Q: Matrix = np.eye(self.dim_x, dtype=float)
        self.chol_Q: Matrix = np.eye(self.dim_x, dtype=float)
        self.R: Matrix = np.eye(self.dim_z, dtype=float)
        self.chol_R: Matrix = np.eye(self.dim_z, dtype=float)

        self.K: Matrix = np.zeros((self.dim_x, self.dim_z), dtype=float)
        self.y: Vector = np.zeros(self.dim_z, dtype=float)
        self.z: Vector = np.zeros(self.dim_z, dtype=float)
        self.S: Matrix = np.eye(self.dim_z, dtype=float)
        self.S_innov: Matrix = np.eye(self.dim_z, dtype=float)

        self.x_prior: Vector = self.x.copy()
        self.P_prior: Matrix = self.P.copy()
        self.x_post: Vector = self.x.copy()
        self.P_post: Matrix = self.P.copy()

        self.z_pred: Vector = np.zeros(self.dim_z, dtype=float)
        self.log_likelihood: float = float("nan")
        self.likelihood: float = float("nan")
        self.mahalanobis: float = float("nan")
        self.nis: float = float("nan")

        # Diagnostics mirror TurboCKF's surface; downdate_fallback_count is
        # specific to the square-root posterior path.
        self.last_jitter: float = 0.0
        self.max_jitter: float = 0.0
        self.jitter_count: int = 0
        self.singular_innovation_count: int = 0
        self.downdate_fallback_count: int = 0

        self._rust_backend = _rust.SquareRootCubatureKalmanFilter(
            self.dim_x, self.dim_z, self.dt
        )
        self._backend_name = "rust-sr"

    def __repr__(self) -> str:
        return (
            f"TurboSRCKF(dim_x={self.dim_x}, dim_z={self.dim_z}, dt={self.dt}, "
            f"backend={self._backend_name!r}, "
            f"log_likelihood={self.log_likelihood:.4g}, "
            f"jitter_count={self.jitter_count}, "
            f"downdate_fallback_count={self.downdate_fallback_count})"
        )

    # ----- prediction ------------------------------------------------------

    def predict(
        self,
        dt: float | None = None,
        fx: Callable[..., npt.ArrayLike] | None = None,
        fx_args: Sequence[object] | object = (),
    ) -> Vector:
        """Time-update step (mirrors ``TurboCKF.predict``)."""

        local_dt = self.dt if dt is None else float(dt)
        if not np.isfinite(local_dt):
            raise ValueError("dt must be finite")
        transition = self.fx if fx is None else fx
        args = TurboCKF._coerce_args(fx_args)

        self._push_state_to_backend()
        self._rust_backend.predict_custom(
            self._make_backend_model(transition, expected_dim=self.dim_x, include_dt=True),
            local_dt,
            args,
        )
        self._pull_state_from_backend()
        return self.x

    # ----- update ----------------------------------------------------------

    def update(
        self,
        z: npt.ArrayLike | None,
        R: npt.ArrayLike | None = None,
        hx: Callable[..., npt.ArrayLike] | None = None,
        hx_args: Sequence[object] | object = (),
    ) -> Vector:
        """Measurement-update step.

        ``z=None`` skips the update and clears innovation diagnostics so the
        next NIS gate can't read a stale value (same contract as
        :meth:`TurboCKF.update`).
        """

        if z is None:
            self._push_state_to_backend()
            self._rust_backend.clear_update_diagnostics()
            self._pull_state_from_backend()
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.z = np.full(self.dim_z, np.nan, dtype=float)
            return self.x

        measurement_fn = self.hx if hx is None else hx
        args = TurboCKF._coerce_args(hx_args)
        z_vec = TurboCKF._as_vector(z, self.dim_z, "z")
        r_mat = TurboCKF._coerce_covariance(self.R if R is None else R, self.dim_z, "R")

        self._push_state_to_backend()
        self._rust_backend.update(
            z_vec,
            self._make_backend_model(measurement_fn, expected_dim=self.dim_z, include_dt=False),
            r_mat,
            args,
        )
        self._pull_state_from_backend()
        return self.x

    def run(
        self,
        zs: npt.ArrayLike | Sequence[npt.ArrayLike | None],
        dts: npt.ArrayLike | None = None,
        Rs: npt.ArrayLike | None = None,
        fx_args: Sequence[object] | object = (),
        hx_args: Sequence[object] | object = (),
        nan_means_missing: bool = False,
    ) -> FilterRun:
        """Run predict+update over a whole measurement sequence in one call.

        Same contract as :meth:`TurboCKF.run` — see that docstring for the
        accepted ``zs``/``dts``/``Rs`` spellings and missed-measurement
        handling.
        """

        return _run_filter(self, zs, dts, Rs, fx_args, hx_args, nan_means_missing)

    # ----- diagnostics / utility ------------------------------------------

    def gate(self, threshold: float) -> bool:
        """Chi-square gating on the most recent NIS (same contract as TurboCKF.gate)."""

        if not np.isfinite(self.nis):
            return False
        return float(self.nis) <= float(threshold)

    def reset(self, x: npt.ArrayLike | None = None, P: npt.ArrayLike | None = None) -> None:
        """Reset state + diagnostics. ``Q``, ``R``, ``dt``, ``fx``, ``hx`` are preserved."""

        self.x = (
            np.zeros(self.dim_x, dtype=float)
            if x is None
            else TurboCKF._as_vector(x, self.dim_x, "x")
        )
        self.P = (
            np.eye(self.dim_x, dtype=float)
            if P is None
            else TurboCKF._coerce_covariance(P, self.dim_x, "P")
        )
        self.K = np.zeros((self.dim_x, self.dim_z), dtype=float)
        self.y = np.zeros(self.dim_z, dtype=float)
        self.z = np.zeros(self.dim_z, dtype=float)
        self.S = np.eye(self.dim_z, dtype=float)
        self.S_innov = np.eye(self.dim_z, dtype=float)
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self.z_pred = np.zeros(self.dim_z, dtype=float)
        self.log_likelihood = float("nan")
        self.likelihood = float("nan")
        self.mahalanobis = float("nan")
        self.nis = float("nan")
        self.last_jitter = 0.0
        self.max_jitter = 0.0
        self.jitter_count = 0
        self.singular_innovation_count = 0
        self.downdate_fallback_count = 0
        self._rust_backend = _rust.SquareRootCubatureKalmanFilter(
            self.dim_x, self.dim_z, self.dt
        )
        self._push_state_to_backend()

    def reset_jitter_counters(self) -> None:
        """Zero all jitter / downdate diagnostics after seeding state.

        Useful for tests that want to measure per-step jitter without the
        one-shot init-time Cholesky cost showing up in the counters.
        """

        self._rust_backend.reset_jitter_counters()
        self.last_jitter = 0.0
        self.max_jitter = 0.0
        self.jitter_count = 0
        self.singular_innovation_count = 0
        self.downdate_fallback_count = 0

    # ----- internals -------------------------------------------------------

    def _push_state_to_backend(self) -> None:
        self._rust_backend.set_state(self._x, self._P, self._Q, self._R)

    def _pull_state_from_backend(self) -> None:
        # Private-slot writes: backend output is already float64 and
        # well-shaped, so the coercing property setters would only add
        # per-step overhead.
        snap = self._rust_backend.snapshot()
        self._x = np.asarray(snap["x"], dtype=float).reshape(-1)
        self.chol_P = np.asarray(snap["chol_P"], dtype=float)
        self._P = np.asarray(snap["P"], dtype=float)
        self.chol_Q = np.asarray(snap["chol_Q"], dtype=float)
        self._Q = np.asarray(snap["Q"], dtype=float)
        self.chol_R = np.asarray(snap["chol_R"], dtype=float)
        self._R = np.asarray(snap["R"], dtype=float)
        self.K = np.asarray(snap["K"], dtype=float)
        self.y = np.asarray(snap["y"], dtype=float).reshape(-1)
        self.z = np.asarray(snap["z"], dtype=float).reshape(-1)
        self.S = np.asarray(snap["S"], dtype=float)
        self.S_innov = np.asarray(snap["S_innov"], dtype=float)
        self.x_prior = np.asarray(snap["x_prior"], dtype=float).reshape(-1)
        self.P_prior = np.asarray(snap["P_prior"], dtype=float)
        self.x_post = np.asarray(snap["x_post"], dtype=float).reshape(-1)
        self.P_post = np.asarray(snap["P_post"], dtype=float)
        self.z_pred = np.asarray(snap["z_pred"], dtype=float).reshape(-1)
        self.log_likelihood = float(snap["log_likelihood"])
        self.likelihood = float(snap["likelihood"])
        self.mahalanobis = float(snap["mahalanobis"])
        self.nis = float(snap["nis"])
        self.last_jitter = float(snap["last_jitter"])
        self.max_jitter = float(snap["max_jitter"])
        self.jitter_count = int(snap["jitter_count"])
        self.singular_innovation_count = int(snap["singular_innovation_count"])
        self.downdate_fallback_count = int(snap["downdate_fallback_count"])

    def _make_backend_model(
        self,
        model: Callable[..., npt.ArrayLike],
        expected_dim: int,
        include_dt: bool,
    ) -> Callable[..., Matrix]:
        def _wrapped(sigma_points, *call_args):
            sigma = np.asarray(sigma_points, dtype=float)
            if sigma.ndim != 2:
                raise ValueError(
                    "fx/hx must accept a 2D batch of sigma points with shape "
                    f"(2 * dim_x, dim_x); got ndim={sigma.ndim}. See README "
                    "for the vectorized-callback contract."
                )
            if include_dt:
                if len(call_args) == 0:
                    raise ValueError("missing dt argument for transition callback")
                local_dt = float(call_args[0])
                extra_args = call_args[1:]
            else:
                local_dt = 0.0
                extra_args = call_args
            args = tuple(extra_args or ())
            expected_shape = (sigma.shape[0], expected_dim)
            if include_dt:
                raw = model(sigma, local_dt, *args)
            else:
                raw = model(sigma, *args)
            arr = np.asarray(raw, dtype=float)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"fx/hx must return shape {expected_shape}, got {arr.shape}. "
                    "TurboSRCKF requires vectorized callbacks — write fx/hx so "
                    "that they map a batch of sigma points (rows) to a batch "
                    "of outputs (rows). Pointwise callbacks are not supported."
                )
            return arr

        return _wrapped

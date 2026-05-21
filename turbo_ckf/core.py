"""Turbo CKF wrapper over the required Rust backend."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

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


class TurboCKF:
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

    def predict_standard_model(self, model_type: str) -> Vector:
        """Predict with the lightweight (KCKF) linear equations."""

        self._validate_standard_model(model_type)
        self._push_state_to_backend()
        self._rust_backend.predict_standard_model(str(model_type))
        self._pull_state_from_backend()
        return self.x

    def predict_standard_model_ckf(self, model_type: str) -> Vector:
        """Predict with the original CKF cubature-point summation equations."""

        self._validate_standard_model(model_type)
        self._push_state_to_backend()
        self._rust_backend.predict_standard_model_ckf(str(model_type))
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
        return self.x

    def update_paper_ahrs(
        self, z: npt.ArrayLike, sigma_acc2: float, sigma_mag2: float
    ) -> Vector:
        """Run Eq. (9), (12)-(14) AHRS update in Rust."""

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

    # ----- diagnostics / utility ------------------------------------------

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
        new._push_state_to_backend()
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> "TurboCKF":
        return self.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the filter state to a plain dict (ndarrays kept as
        ndarrays). Callbacks are *not* included — restoring requires the
        caller to re-supply them via :meth:`from_dict`."""

        return {
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
        kf._push_state_to_backend()
        return kf

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
        self._rust_backend.set_state(self.x, self.P, self.Q, self.R)

    def _pull_state_from_backend(self) -> None:
        # The Rust snapshot already returns fresh numpy buffers (via
        # `ToPyArray::to_pyarray`, which allocates a new PyArray per call),
        # so `np.asarray` here is a zero-copy adoption — no aliasing risk to
        # the backend struct's internal storage.
        snap = self._rust_backend.snapshot()
        self.x = np.asarray(snap["x"], dtype=float).reshape(-1)
        self.P = np.asarray(snap["P"], dtype=float)
        self.Q = np.asarray(snap["Q"], dtype=float)
        self.R = np.asarray(snap["R"], dtype=float)
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

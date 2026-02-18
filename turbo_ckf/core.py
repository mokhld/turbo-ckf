"""Turbo CKF wrapper over the required Rust backend."""

from __future__ import annotations

from typing import Callable, Sequence

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


class TurboCKF:
    """Rust-backed CKF focused on accelerated execution paths."""

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

        self._rust_backend = _rust.CubatureKalmanFilter(self.dim_x, self.dim_z, self.dt)
        self._backend_name = "rust"

    def predict(self, dt=None, fx=None, fx_args=()):
        """Run the time-update step."""

        local_dt = self.dt if dt is None else float(dt)
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

    def predict_standard_model(self, model_type):
        """Predict with the lightweight (KCKF) linear equations."""

        self._push_state_to_backend()
        self._rust_backend.predict_standard_model(str(model_type))
        self._pull_state_from_backend()
        return self.x

    def predict_standard_model_ckf(self, model_type):
        """Predict with the original CKF cubature-point summation equations."""

        self._push_state_to_backend()
        self._rust_backend.predict_standard_model_ckf(str(model_type))
        self._pull_state_from_backend()
        return self.x

    def predict_linear_model(self, f):
        """Predict using KCKF equations with a caller-provided linear transition matrix."""

        f_mat = self._coerce_covariance(f, self.dim_x, "F")
        self._push_state_to_backend()
        self._rust_backend.predict_linear_model(f_mat)
        self._pull_state_from_backend()
        return self.x

    def predict_linear_model_ckf(self, f):
        """Predict using CKF cubature summation equations with a caller-provided linear transition matrix."""

        f_mat = self._coerce_covariance(f, self.dim_x, "F")
        self._push_state_to_backend()
        self._rust_backend.predict_linear_model_ckf(f_mat)
        self._pull_state_from_backend()
        return self.x

    def update(self, z, R=None, hx=None, hx_args=()):
        """Run the measurement-update step."""

        if z is None:
            self.z = np.full(self.dim_z, np.nan, dtype=float)
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
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

    def update_paper_ahrs(self, z, sigma_acc2, sigma_mag2):
        """Run Eq. (9), (12)-(14) AHRS update in Rust."""

        if self.dim_x != 4 or self.dim_z != 6:
            raise ValueError("update_paper_ahrs requires dim_x == 4 and dim_z == 6")
        z_vec = self._as_vector(z, self.dim_z, "z")
        self._push_state_to_backend()
        self._rust_backend.update_paper_ahrs(z_vec, float(sigma_acc2), float(sigma_mag2))
        self._pull_state_from_backend()
        return self.x

    def _push_state_to_backend(self) -> None:
        self._rust_backend.set_state(self.x, self.P, self.Q, self.R)

    def _pull_state_from_backend(self) -> None:
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

    def _make_backend_model(
        self,
        model: Callable[..., npt.ArrayLike],
        expected_dim: int,
        include_dt: bool,
    ) -> Callable[..., Matrix]:
        def _wrapped(sigma_points, *call_args):
            sigma = np.asarray(sigma_points, dtype=float)
            if sigma.ndim != 2:
                raise ValueError(f"sigma points must be a 2D array, got shape {sigma.shape}")
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

    def normalize_state_quaternion(self) -> Vector:
        """Normalize the state when it represents a quaternion."""

        if self.dim_x != 4:
            raise ValueError("normalize_state_quaternion requires dim_x == 4")
        norm = float(np.linalg.norm(self.x))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("quaternion norm must be finite and positive")
        self.x = self.x / norm
        return self.x

    def normalize_state_quaternion_backend(self) -> Vector:
        """Normalize quaternion state directly inside the Rust backend."""

        if self.dim_x != 4:
            raise ValueError("normalize_state_quaternion_backend requires dim_x == 4")
        self._push_state_to_backend()
        self._rust_backend.normalize_quaternion_state()
        self._pull_state_from_backend()
        return self.x

    def _cubature_points(self, mean: npt.ArrayLike, cov: npt.ArrayLike) -> Matrix:
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

    def _apply_measurement(self, hx: Callable[..., npt.ArrayLike], sigma_points: Matrix, hx_args: Sequence[object]) -> Matrix:
        return self._apply_model(
            model=hx,
            sigma_points=sigma_points,
            expected_dim=self.dim_z,
            include_dt=False,
            dt=0.0,
            extra_args=hx_args,
        )

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
            raise ValueError(f"model output must have shape {expected_shape}, got {arr.shape}")
        return arr

    def _stable_cholesky(self, cov: Matrix) -> Matrix:
        jitter = 0.0
        eye = np.eye(cov.shape[0], dtype=float)
        for _ in range(6):
            try:
                return np.linalg.cholesky(cov + jitter * eye)
            except np.linalg.LinAlgError:
                jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
        return np.linalg.cholesky(cov + 1e-6 * eye)

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
        if args is None:
            return ()
        if isinstance(args, tuple):
            return args
        return (args,)

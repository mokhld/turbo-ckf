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
        per-step arrays with leading dimension ``N``. Per-step inputs
        match the contract that ``rts_smooth`` consumes, so a
        forward-then-backward pass is one composed call away.

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
            Fs: per-step transition matrices ``F_k`` that map step ``k`` to
                ``k+1``. Shape ``(N, dim_x, dim_x)`` (FilterPy-compatible —
                last entry unused) or ``(N-1, dim_x, dim_x)``.
            Qs: per-step process-noise covariances ``Q_k``. Same shape rules
                as ``Fs``.

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


class TurboSRCKF:
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
        self._rust_backend.set_state(self.x, self.P, self.Q, self.R)

    def _pull_state_from_backend(self) -> None:
        snap = self._rust_backend.snapshot()
        self.x = np.asarray(snap["x"], dtype=float).reshape(-1)
        self.chol_P = np.asarray(snap["chol_P"], dtype=float)
        self.P = np.asarray(snap["P"], dtype=float)
        self.chol_Q = np.asarray(snap["chol_Q"], dtype=float)
        self.Q = np.asarray(snap["Q"], dtype=float)
        self.chol_R = np.asarray(snap["chol_R"], dtype=float)
        self.R = np.asarray(snap["R"], dtype=float)
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

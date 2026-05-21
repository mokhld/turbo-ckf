//! Turbo CKF Rust backend.
//!
//! PyO3 0.20's `#[pymethods]` macro emits non-local `impl` blocks. That is
//! a lint introduced in newer rustc versions; bumping PyO3 fixes it but is
//! out of scope for this change.
#![allow(non_local_definitions)]

use nalgebra::{DMatrix, DVector};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::f64::consts::PI;

#[pyclass]
pub struct CubatureKalmanFilter {
    dim_x: usize,
    dim_z: usize,
    dt: f64,
    x: DVector<f64>,
    p: DMatrix<f64>,
    q: DMatrix<f64>,
    r: DMatrix<f64>,
    k: DMatrix<f64>,
    y: DVector<f64>,
    z: DVector<f64>,
    s: DMatrix<f64>,
    si: DMatrix<f64>,
    x_prior: DVector<f64>,
    p_prior: DMatrix<f64>,
    x_post: DVector<f64>,
    p_post: DMatrix<f64>,
    z_pred: DVector<f64>,
    log_likelihood: f64,
    likelihood: f64,
    mahalanobis: f64,
    nis: f64,
    // Diagnostics for stable_cholesky: jitter added to the diagonal because
    // P drifted toward non-positive-definite. Surfaced via snapshot() so users
    // can detect silent numerical conditioning issues.
    last_jitter: f64,
    max_jitter: f64,
    jitter_count: u64,
    singular_innovation_count: u64,
}

#[pymethods]
impl CubatureKalmanFilter {
    #[new]
    fn new(dim_x: usize, dim_z: usize, dt: f64) -> PyResult<Self> {
        if dim_x == 0 || dim_z == 0 {
            return Err(PyValueError::new_err("dim_x and dim_z must be positive"));
        }
        if !dt.is_finite() {
            return Err(PyValueError::new_err("dt must be finite"));
        }
        Ok(Self {
            dim_x,
            dim_z,
            dt,
            x: DVector::zeros(dim_x),
            p: DMatrix::identity(dim_x, dim_x),
            q: DMatrix::identity(dim_x, dim_x),
            r: DMatrix::identity(dim_z, dim_z),
            k: DMatrix::zeros(dim_x, dim_z),
            y: DVector::zeros(dim_z),
            z: DVector::zeros(dim_z),
            s: DMatrix::identity(dim_z, dim_z),
            si: DMatrix::identity(dim_z, dim_z),
            x_prior: DVector::zeros(dim_x),
            p_prior: DMatrix::identity(dim_x, dim_x),
            x_post: DVector::zeros(dim_x),
            p_post: DMatrix::identity(dim_x, dim_x),
            z_pred: DVector::zeros(dim_z),
            log_likelihood: f64::NAN,
            likelihood: f64::NAN,
            mahalanobis: f64::NAN,
            nis: f64::NAN,
            last_jitter: 0.0,
            max_jitter: 0.0,
            jitter_count: 0,
            singular_innovation_count: 0,
        })
    }

    #[pyo3(signature = (x, p, q, r))]
    fn set_state(
        &mut self,
        x: PyReadonlyArray1<f64>,
        p: PyReadonlyArray2<f64>,
        q: PyReadonlyArray2<f64>,
        r: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.x = pyarray1_to_dvector(x, self.dim_x, "x")?;
        self.p = pyarray2_to_dmatrix(p, self.dim_x, self.dim_x, "P")?;
        self.q = pyarray2_to_dmatrix(q, self.dim_x, self.dim_x, "Q")?;
        self.r = pyarray2_to_dmatrix(r, self.dim_z, self.dim_z, "R")?;
        Ok(())
    }

    #[pyo3(signature = (fx, dt=None, fx_args=None))]
    fn predict_custom(
        &mut self,
        py: Python<'_>,
        fx: PyObject,
        dt: Option<f64>,
        fx_args: Option<&PyTuple>,
    ) -> PyResult<()> {
        let local_dt = dt.unwrap_or(self.dt);
        if !local_dt.is_finite() {
            return Err(PyValueError::new_err("dt must be finite"));
        }
        let (sigma, jitter) = cubature_points(&self.x, &self.p)?;
        self.record_jitter(jitter);
        let args = fx_args.unwrap_or_else(|| PyTuple::empty(py));
        let propagated = call_model_vectorized(py, &fx, &sigma, Some(local_dt), args, self.dim_x)?;

        let w = 1.0 / (propagated.nrows() as f64);

        let mean = row_mean(&propagated, self.dim_x);
        // tr_mul avoids materializing propagated.transpose().
        let second_moment = propagated.tr_mul(&propagated) * w;
        let mut cov = second_moment - &mean * mean.transpose() + &self.q;
        symmetrize_in_place(&mut cov);

        self.x = mean;
        self.p = cov;
        self.x_prior = self.x.clone();
        self.p_prior = self.p.clone();
        Ok(())
    }

    #[pyo3(signature = (model_type))]
    fn predict_standard_model(&mut self, model_type: &str) -> PyResult<()> {
        let f = transition_matrix(model_type, self.dim_x, self.dt)?;
        self.predict_kckf_linear(&f);
        Ok(())
    }

    /// Convenience for callers: list of model strings accepted by
    /// predict_standard_model[_ckf]. Lets the Python wrapper validate input
    /// with a friendly error before reaching the backend.
    #[staticmethod]
    fn supported_standard_models() -> Vec<&'static str> {
        vec!["constant_velocity", "constant_acceleration"]
    }

    #[pyo3(signature = (model_type))]
    fn predict_standard_model_ckf(&mut self, model_type: &str) -> PyResult<()> {
        let f = transition_matrix(model_type, self.dim_x, self.dt)?;
        self.predict_ckf_linear(&f)?;
        Ok(())
    }

    #[pyo3(signature = (f))]
    fn predict_linear_model(&mut self, f: PyReadonlyArray2<f64>) -> PyResult<()> {
        let f_mat = pyarray2_to_dmatrix(f, self.dim_x, self.dim_x, "F")?;
        self.predict_kckf_linear(&f_mat);
        Ok(())
    }

    #[pyo3(signature = (f))]
    fn predict_linear_model_ckf(&mut self, f: PyReadonlyArray2<f64>) -> PyResult<()> {
        let f_mat = pyarray2_to_dmatrix(f, self.dim_x, self.dim_x, "F")?;
        self.predict_ckf_linear(&f_mat)?;
        Ok(())
    }

    /// Clear cached innovation / likelihood diagnostics. Used by the Python
    /// wrapper when a measurement is skipped (`update(z=None)`), so callers
    /// don't accidentally re-read stale values.
    fn clear_update_diagnostics(&mut self) {
        self.y = DVector::zeros(self.dim_z);
        self.z = DVector::zeros(self.dim_z);
        self.z_pred = DVector::zeros(self.dim_z);
        self.s = DMatrix::identity(self.dim_z, self.dim_z);
        self.si = DMatrix::identity(self.dim_z, self.dim_z);
        self.k = DMatrix::zeros(self.dim_x, self.dim_z);
        self.log_likelihood = f64::NAN;
        self.likelihood = f64::NAN;
        self.mahalanobis = f64::NAN;
        self.nis = f64::NAN;
    }

    fn reset_jitter_counters(&mut self) {
        self.last_jitter = 0.0;
        self.max_jitter = 0.0;
        self.jitter_count = 0;
        self.singular_innovation_count = 0;
    }

    #[pyo3(signature = (z, hx, r=None, hx_args=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        hx: PyObject,
        r: Option<PyReadonlyArray2<f64>>,
        hx_args: Option<&PyTuple>,
    ) -> PyResult<()> {
        let z_vec = pyarray1_to_dvector(z, self.dim_z, "z")?;
        let r_mat = if let Some(mat) = r {
            pyarray2_to_dmatrix(mat, self.dim_z, self.dim_z, "R")?
        } else {
            self.r.clone()
        };

        let (sigma, jitter) = cubature_points(&self.x, &self.p)?;
        self.record_jitter(jitter);
        let args = hx_args.unwrap_or_else(|| PyTuple::empty(py));
        let z_sigma = call_model_vectorized(py, &hx, &sigma, None, args, self.dim_z)?;

        let w = 1.0 / (sigma.nrows() as f64);

        let z_pred = row_mean(&z_sigma, self.dim_z);
        let pxz = sigma.tr_mul(&z_sigma) * w - (&self.x * z_pred.transpose());
        let mut pzz = z_sigma.tr_mul(&z_sigma) * w - &z_pred * z_pred.transpose() + &r_mat;
        symmetrize_in_place(&mut pzz);

        let (si, was_singular) = invert_innovation(&pzz)?;
        if was_singular {
            self.singular_innovation_count = self.singular_innovation_count.saturating_add(1);
        }

        let k = &pxz * &si;
        let y = &z_vec - &z_pred;

        self.x = &self.x + &k * &y;
        let mut p_new = &self.p - &k * &pzz * k.transpose();
        symmetrize_in_place(&mut p_new);
        self.p = p_new;

        self.z = z_vec;
        self.z_pred = z_pred;
        self.k = k;
        self.y = y;
        self.s = pzz;
        self.si = si;
        self.x_post = self.x.clone();
        self.p_post = self.p.clone();
        self.update_likelihood_terms();
        Ok(())
    }

    #[pyo3(signature = (z, sigma_acc2, sigma_mag2))]
    fn update_paper_ahrs(
        &mut self,
        z: PyReadonlyArray1<f64>,
        sigma_acc2: f64,
        sigma_mag2: f64,
    ) -> PyResult<()> {
        if self.dim_x != 4 || self.dim_z != 6 {
            return Err(PyValueError::new_err(
                "update_paper_ahrs requires dim_x == 4 and dim_z == 6",
            ));
        }
        if !sigma_acc2.is_finite() || sigma_acc2 <= 0.0 {
            return Err(PyValueError::new_err(
                "sigma_acc2 must be finite and positive",
            ));
        }
        if !sigma_mag2.is_finite() || sigma_mag2 <= 0.0 {
            return Err(PyValueError::new_err(
                "sigma_mag2 must be finite and positive",
            ));
        }
        let z_vec = pyarray1_to_dvector(z, self.dim_z, "z")?;
        let (m_n, m_d) = magnetic_reference_terms(&z_vec)?;

        let (sigma, jitter) = cubature_points(&self.x, &self.p)?;
        self.record_jitter(jitter);
        let z_sigma = paper_observation_model(&sigma, m_n, m_d);

        let mut r_mat = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            r_mat[(i, i)] = sigma_acc2;
            r_mat[(i + 3, i + 3)] = sigma_mag2;
        }

        let w = 1.0 / (sigma.nrows() as f64);

        let z_pred = row_mean(&z_sigma, self.dim_z);
        let pxz = sigma.tr_mul(&z_sigma) * w - (&self.x * z_pred.transpose());
        let mut pzz = z_sigma.tr_mul(&z_sigma) * w - &z_pred * z_pred.transpose() + &r_mat;
        symmetrize_in_place(&mut pzz);

        let (si, was_singular) = invert_innovation(&pzz)?;
        if was_singular {
            self.singular_innovation_count = self.singular_innovation_count.saturating_add(1);
        }

        let k = &pxz * &si;
        let y = &z_vec - &z_pred;

        self.x = &self.x + &k * &y;
        let mut p_new = &self.p - &k * &pzz * k.transpose();
        symmetrize_in_place(&mut p_new);
        self.p = p_new;

        self.z = z_vec;
        self.z_pred = z_pred;
        self.k = k;
        self.y = y;
        self.s = pzz;
        self.si = si;
        self.r = r_mat;
        self.x_post = self.x.clone();
        self.p_post = self.p.clone();
        self.update_likelihood_terms();
        Ok(())
    }

    fn normalize_quaternion_state(&mut self) -> PyResult<()> {
        if self.dim_x != 4 {
            return Err(PyValueError::new_err(
                "normalize_quaternion_state requires dim_x == 4",
            ));
        }
        let norm = self.x.norm();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(PyValueError::new_err(
                "quaternion norm must be finite and positive",
            ));
        }
        self.x /= norm;
        self.x_post = self.x.clone();
        Ok(())
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("x", self.x.as_slice().to_pyarray(py))?;
        out.set_item("P", dmatrix_to_pyarray(py, &self.p)?)?;
        out.set_item("Q", dmatrix_to_pyarray(py, &self.q)?)?;
        out.set_item("R", dmatrix_to_pyarray(py, &self.r)?)?;
        out.set_item("K", dmatrix_to_pyarray(py, &self.k)?)?;
        out.set_item("y", self.y.as_slice().to_pyarray(py))?;
        out.set_item("z", self.z.as_slice().to_pyarray(py))?;
        out.set_item("S", dmatrix_to_pyarray(py, &self.s)?)?;
        out.set_item("SI", dmatrix_to_pyarray(py, &self.si)?)?;
        out.set_item("x_prior", self.x_prior.as_slice().to_pyarray(py))?;
        out.set_item("P_prior", dmatrix_to_pyarray(py, &self.p_prior)?)?;
        out.set_item("x_post", self.x_post.as_slice().to_pyarray(py))?;
        out.set_item("P_post", dmatrix_to_pyarray(py, &self.p_post)?)?;
        out.set_item("z_pred", self.z_pred.as_slice().to_pyarray(py))?;
        out.set_item("log_likelihood", self.log_likelihood)?;
        out.set_item("likelihood", self.likelihood)?;
        out.set_item("mahalanobis", self.mahalanobis)?;
        out.set_item("nis", self.nis)?;
        out.set_item("last_jitter", self.last_jitter)?;
        out.set_item("max_jitter", self.max_jitter)?;
        out.set_item("jitter_count", self.jitter_count)?;
        out.set_item("singular_innovation_count", self.singular_innovation_count)?;
        Ok(out.to_object(py))
    }
}

impl CubatureKalmanFilter {
    fn predict_kckf_linear(&mut self, f: &DMatrix<f64>) {
        self.x = f * &self.x;
        let mut new_p = f * &self.p * f.transpose() + &self.q;
        symmetrize_in_place(&mut new_p);
        self.p = new_p;
        self.x_prior = self.x.clone();
        self.p_prior = self.p.clone();
    }

    fn predict_ckf_linear(&mut self, f: &DMatrix<f64>) -> PyResult<()> {
        let (sigma, jitter) = cubature_points(&self.x, &self.p)?;
        self.record_jitter(jitter);
        // Propagate each row-vector cubature point through the linear model.
        let propagated = &sigma * f.transpose();
        let w = 1.0 / (propagated.nrows() as f64);

        let mean = row_mean(&propagated, self.dim_x);
        let second_moment = propagated.tr_mul(&propagated) * w;
        let mut cov = second_moment - &mean * mean.transpose() + &self.q;
        symmetrize_in_place(&mut cov);

        self.x = mean;
        self.p = cov;
        self.x_prior = self.x.clone();
        self.p_prior = self.p.clone();
        Ok(())
    }

    /// Compute log-likelihood using the Cholesky factor of S. This is both
    /// faster than `lu().determinant()` and more numerically robust: if S is
    /// not positive-definite (which can happen after many updates if R is
    /// tiny relative to drift), we bail to a clear `-inf` rather than risk
    /// a near-zero or negative determinant.
    fn update_likelihood_terms(&mut self) {
        let mahal2 = (self.y.transpose() * &self.si * &self.y)[(0, 0)];
        self.nis = mahal2.max(0.0);
        self.mahalanobis = self.nis.sqrt();

        if let Some(chol) = self.s.clone().cholesky() {
            // log det(S) = 2 * sum(log(diag(L)))
            let l = chol.l();
            let mut log_det = 0.0;
            for i in 0..l.nrows() {
                let d = l[(i, i)];
                if d <= 0.0 || !d.is_finite() {
                    self.log_likelihood = f64::NEG_INFINITY;
                    self.likelihood = 0.0;
                    return;
                }
                log_det += d.ln();
            }
            log_det *= 2.0;
            self.log_likelihood = -0.5 * ((self.dim_z as f64) * (2.0 * PI).ln() + log_det + mahal2);
            self.likelihood = self.log_likelihood.exp();
        } else {
            self.log_likelihood = f64::NEG_INFINITY;
            self.likelihood = 0.0;
        }
    }

    #[inline]
    fn record_jitter(&mut self, jitter: f64) {
        self.last_jitter = jitter;
        if jitter > 0.0 {
            self.jitter_count = self.jitter_count.saturating_add(1);
            if jitter > self.max_jitter {
                self.max_jitter = jitter;
            }
        }
    }
}

fn call_model_vectorized(
    py: Python<'_>,
    model: &PyObject,
    sigma: &DMatrix<f64>,
    dt: Option<f64>,
    extra_args: &PyTuple,
    expected_dim: usize,
) -> PyResult<DMatrix<f64>> {
    let sigma_arr = dmatrix_to_pyarray(py, sigma)?;

    let mut args: Vec<PyObject> = Vec::with_capacity(2 + extra_args.len());
    args.push(sigma_arr.to_object(py));
    if let Some(value) = dt {
        args.push(value.into_py(py));
    }
    for item in extra_args.iter() {
        args.push(item.to_object(py));
    }
    let py_args = PyTuple::new(py, args);
    let out_obj = model.call1(py, py_args)?;
    let out_arr: PyReadonlyArray2<f64> = out_obj.extract(py)?;

    let out = pyarray2_to_dmatrix(out_arr, sigma.nrows(), expected_dim, "model output")?;
    Ok(out)
}

fn transition_matrix(model_type: &str, dim_x: usize, dt: f64) -> PyResult<DMatrix<f64>> {
    match model_type {
        "constant_velocity" => {
            if !dim_x.is_multiple_of(2) {
                return Err(PyValueError::new_err(
                    "constant_velocity requires even dim_x with [pos..., vel...] layout",
                ));
            }
            let n = dim_x / 2;
            let mut f = DMatrix::<f64>::identity(dim_x, dim_x);
            for i in 0..n {
                f[(i, n + i)] = dt;
            }
            Ok(f)
        }
        "constant_acceleration" => {
            if !dim_x.is_multiple_of(3) {
                return Err(PyValueError::new_err(
                    "constant_acceleration requires dim_x multiple of 3 with [pos..., vel..., acc...] layout",
                ));
            }
            let n = dim_x / 3;
            let mut f = DMatrix::<f64>::identity(dim_x, dim_x);
            for i in 0..n {
                f[(i, n + i)] = dt;
                f[(i, 2 * n + i)] = 0.5 * dt * dt;
                f[(n + i, 2 * n + i)] = dt;
            }
            Ok(f)
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported model_type: {model_type}"
        ))),
    }
}

#[inline]
fn cubature_points(x: &DVector<f64>, p: &DMatrix<f64>) -> PyResult<(DMatrix<f64>, f64)> {
    let n = x.nrows();
    let (chol, applied_jitter) = stable_cholesky(p)?;
    let scale = (n as f64).sqrt();

    let mut sigma = DMatrix::<f64>::zeros(2 * n, n);
    for k in 0..n {
        for j in 0..n {
            // chol is the lower-triangular L from p.cholesky(). Indexing
            // chol[(j, k)] effectively reads the column of L^T, which is the
            // convention used by FilterPy's scipy-based reference.
            let offset = scale * chol[(j, k)];
            sigma[(k, j)] = x[j] + offset;
            sigma[(n + k, j)] = x[j] - offset;
        }
    }
    Ok((sigma, applied_jitter))
}

/// Returns the Cholesky factor of `p` plus any jitter that had to be added to
/// the diagonal to make the decomposition succeed. Callers should record the
/// jitter so the user can see when P is silently being conditioned.
fn stable_cholesky(p: &DMatrix<f64>) -> PyResult<(DMatrix<f64>, f64)> {
    let n = p.nrows();
    let eye = DMatrix::<f64>::identity(n, n);
    let mut jitter = 0.0_f64;

    for _ in 0..8 {
        let candidate = p + eye.scale(jitter);
        if let Some(chol) = candidate.cholesky() {
            return Ok((chol.l(), jitter));
        }
        jitter = if jitter == 0.0 { 1e-12 } else { jitter * 10.0 };
    }
    Err(PyRuntimeError::new_err(
        "unable to compute stable Cholesky factor",
    ))
}

fn magnetic_reference_terms(z: &DVector<f64>) -> PyResult<(f64, f64)> {
    let ax = z[0];
    let ay = z[1];
    let az = z[2];
    let mx = z[3];
    let my = z[4];
    let mz = z[5];

    let acc_norm = (ax * ax + ay * ay + az * az).sqrt();
    let mag_norm = (mx * mx + my * my + mz * mz).sqrt();
    if !acc_norm.is_finite() || !mag_norm.is_finite() || acc_norm <= 0.0 || mag_norm <= 0.0 {
        return Err(PyValueError::new_err(
            "acceleration and magnetic measurements must have positive finite norms",
        ));
    }

    let mut m_d = (ax * mx + ay * my + az * mz) / (acc_norm * mag_norm);
    m_d = m_d.clamp(-1.0, 1.0);
    let m_n = (1.0 - m_d * m_d).max(0.0).sqrt();
    Ok((m_n, m_d))
}

fn paper_observation_model(sigma: &DMatrix<f64>, m_n: f64, m_d: f64) -> DMatrix<f64> {
    let mut out = DMatrix::<f64>::zeros(sigma.nrows(), 6);
    for i in 0..sigma.nrows() {
        let q0 = sigma[(i, 0)];
        let q1 = sigma[(i, 1)];
        let q2 = sigma[(i, 2)];
        let q3 = sigma[(i, 3)];

        out[(i, 0)] = 2.0 * (q1 * q3 - q0 * q2);
        out[(i, 1)] = 2.0 * (q2 * q3 + q0 * q1);
        out[(i, 2)] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
        out[(i, 3)] =
            (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * m_n + 2.0 * (q1 * q3 - q0 * q2) * m_d;
        out[(i, 4)] = 2.0 * (q1 * q2 - q0 * q3) * m_n + 2.0 * (q2 * q3 + q0 * q1) * m_d;
        out[(i, 5)] =
            2.0 * (q1 * q3 + q0 * q2) * m_n + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * m_d;
    }
    out
}

#[inline]
fn row_mean(values: &DMatrix<f64>, dim: usize) -> DVector<f64> {
    let w = 1.0 / (values.nrows() as f64);
    let mut mean = DVector::<f64>::zeros(dim);
    // values is (M, dim); sum each column then scale.
    for j in 0..dim {
        let mut acc = 0.0;
        for i in 0..values.nrows() {
            acc += values[(i, j)];
        }
        mean[j] = acc * w;
    }
    mean
}

#[inline]
fn symmetrize_in_place(mat: &mut DMatrix<f64>) {
    let n = mat.nrows();
    debug_assert_eq!(
        n,
        mat.ncols(),
        "symmetrize_in_place requires a square matrix"
    );
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (mat[(i, j)] + mat[(j, i)]);
            mat[(i, j)] = avg;
            mat[(j, i)] = avg;
        }
    }
}

/// Invert the innovation covariance S. Returns (S^-1, was_singular). When S
/// is singular we fall back to the Moore-Penrose pseudo-inverse; callers
/// should bump a counter so the user can see this happened.
fn invert_innovation(s: &DMatrix<f64>) -> PyResult<(DMatrix<f64>, bool)> {
    if let Some(chol) = s.clone().cholesky() {
        return Ok((chol.inverse(), false));
    }
    if let Some(inv) = s.clone().try_inverse() {
        return Ok((inv, true));
    }
    let pinv = s
        .clone()
        .svd(true, true)
        .pseudo_inverse(1e-12)
        .map_err(|_| PyRuntimeError::new_err("failed to invert innovation covariance"))?;
    Ok((pinv, true))
}

fn pyarray1_to_dvector(
    arr: PyReadonlyArray1<f64>,
    expected: usize,
    name: &str,
) -> PyResult<DVector<f64>> {
    if arr.shape()[0] != expected {
        return Err(PyValueError::new_err(format!(
            "{name} must have length {expected}, got {}",
            arr.shape()[0]
        )));
    }
    let view = arr.as_slice()?;
    Ok(DVector::from_column_slice(view))
}

fn pyarray2_to_dmatrix(
    arr: PyReadonlyArray2<f64>,
    expected_rows: usize,
    expected_cols: usize,
    name: &str,
) -> PyResult<DMatrix<f64>> {
    let shape = arr.shape();
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({expected_rows}, {expected_cols}), got ({}, {})",
            shape[0], shape[1]
        )));
    }
    let data = arr.as_array();
    // ndarray default iteration is row-major; nalgebra needs row-major
    // input via `from_row_iterator`. Avoids the previous (i,j) double loop.
    Ok(DMatrix::<f64>::from_row_iterator(
        expected_rows,
        expected_cols,
        data.iter().copied(),
    ))
}

fn dmatrix_to_pyarray<'py>(py: Python<'py>, mat: &DMatrix<f64>) -> PyResult<&'py PyArray2<f64>> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    // Build a single Vec in row-major order (no per-row allocations) then
    // hand it to ndarray::Array2::from_shape_vec; ToPyArray copies into a
    // fresh numpy buffer. One alloc instead of `rows + 1`.
    let mut data: Vec<f64> = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(mat[(i, j)]);
        }
    }
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|_| PyRuntimeError::new_err("failed to build numpy matrix shape"))?;
    Ok(arr.to_pyarray(py))
}

/// Rauch-Tung-Striebel fixed-interval smoother.
///
/// Inputs are the forward-filtered trace:
///   xs: (N, dim_x)              -- filtered state means x_{k|k}
///   ps: (N, dim_x, dim_x)       -- filtered covariances P_{k|k}
///   fs: (N or N-1, dim_x, dim_x) -- per-step transition F_k (maps k -> k+1)
///   qs: (N or N-1, dim_x, dim_x) -- per-step process-noise Q_k
///
/// When fs/qs are passed with length N (FilterPy convention) the last entry
/// is unused. Per-step matrices let the smoother handle non-constant
/// transitions without callbacks.
///
/// Returns (xs_smooth, Ps_smooth) of the same shapes as (xs, ps). The inverse
/// of the one-step-predicted covariance uses the same Cholesky-with-fallback
/// path as the filter's innovation inversion, so smoothing won't blow up on
/// a numerically borderline predicted covariance.
#[pyfunction]
fn rts_smooth<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray2<f64>,
    ps: PyReadonlyArray3<f64>,
    fs: PyReadonlyArray3<f64>,
    qs: PyReadonlyArray3<f64>,
) -> PyResult<(&'py PyArray2<f64>, &'py PyArray3<f64>)> {
    let xs_arr = xs.as_array();
    let ps_arr = ps.as_array();
    let fs_arr = fs.as_array();
    let qs_arr = qs.as_array();

    let n = xs_arr.shape()[0];
    let dim_x = xs_arr.shape()[1];
    if n == 0 {
        return Err(PyValueError::new_err(
            "xs must contain at least one filtered state",
        ));
    }
    if dim_x == 0 {
        return Err(PyValueError::new_err(
            "xs second dimension must be positive",
        ));
    }
    let p_shape = ps_arr.shape();
    if p_shape[0] != n || p_shape[1] != dim_x || p_shape[2] != dim_x {
        return Err(PyValueError::new_err(format!(
            "Ps must have shape ({n}, {dim_x}, {dim_x}), got ({}, {}, {})",
            p_shape[0], p_shape[1], p_shape[2]
        )));
    }
    let need_steps = n.saturating_sub(1);
    for (name, a) in [("Fs", fs_arr.shape()), ("Qs", qs_arr.shape())] {
        let len_ok = a[0] == n || a[0] == need_steps;
        if !len_ok || a[1] != dim_x || a[2] != dim_x {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (N, {dim_x}, {dim_x}) or (N-1, {dim_x}, {dim_x}); got ({}, {}, {})",
                a[0], a[1], a[2]
            )));
        }
    }

    // Pull traces into nalgebra once.
    let mut xs_smooth: Vec<DVector<f64>> = (0..n)
        .map(|i| DVector::from_iterator(dim_x, (0..dim_x).map(|j| xs_arr[[i, j]])))
        .collect();
    let mut ps_smooth: Vec<DMatrix<f64>> = (0..n)
        .map(|i| DMatrix::from_fn(dim_x, dim_x, |r, c| ps_arr[[i, r, c]]))
        .collect();

    // Snapshot filtered values (we overwrite xs_smooth / ps_smooth backwards).
    let xs_filt: Vec<DVector<f64>> = xs_smooth.clone();
    let ps_filt: Vec<DMatrix<f64>> = ps_smooth.clone();

    if n >= 2 {
        for k in (0..n - 1).rev() {
            let f_k = DMatrix::from_fn(dim_x, dim_x, |r, c| fs_arr[[k, r, c]]);
            let q_k = DMatrix::from_fn(dim_x, dim_x, |r, c| qs_arr[[k, r, c]]);

            let x_filt_k = &xs_filt[k];
            let p_filt_k = &ps_filt[k];

            let x_pred = &f_k * x_filt_k;
            let mut p_pred = &f_k * p_filt_k * f_k.transpose() + &q_k;
            symmetrize_in_place(&mut p_pred);

            let (p_pred_inv, _was_singular) = invert_innovation(&p_pred)?;
            let c_k = p_filt_k * f_k.transpose() * &p_pred_inv;

            let x_next_smooth = &xs_smooth[k + 1];
            let p_next_smooth = &ps_smooth[k + 1];

            let x_new = x_filt_k + &c_k * (x_next_smooth - &x_pred);
            let mut p_new = p_filt_k + &c_k * (p_next_smooth - &p_pred) * c_k.transpose();
            symmetrize_in_place(&mut p_new);

            xs_smooth[k] = x_new;
            ps_smooth[k] = p_new;
        }
    }

    // Pack back into numpy buffers (one allocation per output).
    let mut xs_out = numpy::ndarray::Array2::<f64>::zeros((n, dim_x));
    let mut ps_out = numpy::ndarray::Array3::<f64>::zeros((n, dim_x, dim_x));
    for i in 0..n {
        for j in 0..dim_x {
            xs_out[[i, j]] = xs_smooth[i][j];
        }
        for r in 0..dim_x {
            for c in 0..dim_x {
                ps_out[[i, r, c]] = ps_smooth[i][(r, c)];
            }
        }
    }
    Ok((xs_out.to_pyarray(py), ps_out.to_pyarray(py)))
}

/// Linear Kalman batch filter — single pass over an observation sequence
/// with all state and matrices held in Rust.
///
/// Inputs (all already broadcast to length N by the Python wrapper):
///   x0: (dim_x,)
///   p0: (dim_x, dim_x)
///   zs: (N, dim_z)
///   fs: (N, dim_x, dim_x) -- F_k for the predict step k-1 -> k
///   hs: (N, dim_z, dim_x) -- H_k for the linear measurement at step k
///   qs: (N, dim_x, dim_x)
///   rs: (N, dim_z, dim_z)
///
/// Returns (xs, Ps, log_likelihoods) with shapes (N, dim_x),
/// (N, dim_x, dim_x), (N,). The output (xs, Ps) is directly consumable by
/// rts_smooth; no extra reshape needed.
///
/// Posterior update uses Joseph form
/// (P = (I - K H) P (I - K H)^T + K R K^T) to keep P PSD over long runs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn batch_filter_linear<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<f64>,
    p0: PyReadonlyArray2<f64>,
    zs: PyReadonlyArray2<f64>,
    fs: PyReadonlyArray3<f64>,
    hs: PyReadonlyArray3<f64>,
    qs: PyReadonlyArray3<f64>,
    rs: PyReadonlyArray3<f64>,
) -> PyResult<(&'py PyArray2<f64>, &'py PyArray3<f64>, &'py PyArray1<f64>)> {
    let x0_arr = x0.as_array();
    let p0_arr = p0.as_array();
    let zs_arr = zs.as_array();
    let fs_arr = fs.as_array();
    let hs_arr = hs.as_array();
    let qs_arr = qs.as_array();
    let rs_arr = rs.as_array();

    let dim_x = x0_arr.shape()[0];
    if dim_x == 0 {
        return Err(PyValueError::new_err("x0 must have at least one dimension"));
    }
    let n = zs_arr.shape()[0];
    if n == 0 {
        return Err(PyValueError::new_err(
            "zs must contain at least one observation",
        ));
    }
    let dim_z = zs_arr.shape()[1];

    let p0_shape = p0_arr.shape();
    if p0_shape[0] != dim_x || p0_shape[1] != dim_x {
        return Err(PyValueError::new_err(format!(
            "P0 must have shape ({dim_x}, {dim_x}); got ({}, {})",
            p0_shape[0], p0_shape[1]
        )));
    }
    let check_3d = |name: &str, arr: &[usize], a: usize, b: usize, c: usize| -> PyResult<()> {
        if arr[0] != a || arr[1] != b || arr[2] != c {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({a}, {b}, {c}); got ({}, {}, {})",
                arr[0], arr[1], arr[2]
            )));
        }
        Ok(())
    };
    check_3d("Fs", fs_arr.shape(), n, dim_x, dim_x)?;
    check_3d("Hs", hs_arr.shape(), n, dim_z, dim_x)?;
    check_3d("Qs", qs_arr.shape(), n, dim_x, dim_x)?;
    check_3d("Rs", rs_arr.shape(), n, dim_z, dim_z)?;

    let mut x = DVector::from_iterator(dim_x, (0..dim_x).map(|j| x0_arr[j]));
    let mut p = DMatrix::from_fn(dim_x, dim_x, |r, c| p0_arr[[r, c]]);

    let mut xs_out = numpy::ndarray::Array2::<f64>::zeros((n, dim_x));
    let mut ps_out = numpy::ndarray::Array3::<f64>::zeros((n, dim_x, dim_x));
    let mut ll_out = numpy::ndarray::Array1::<f64>::zeros(n);

    let eye_x = DMatrix::<f64>::identity(dim_x, dim_x);
    let log_two_pi = (2.0 * PI).ln();

    for k in 0..n {
        let f_k = DMatrix::from_fn(dim_x, dim_x, |r, c| fs_arr[[k, r, c]]);
        let h_k = DMatrix::from_fn(dim_z, dim_x, |r, c| hs_arr[[k, r, c]]);
        let q_k = DMatrix::from_fn(dim_x, dim_x, |r, c| qs_arr[[k, r, c]]);
        let r_k = DMatrix::from_fn(dim_z, dim_z, |r, c| rs_arr[[k, r, c]]);
        let z_k = DVector::from_iterator(dim_z, (0..dim_z).map(|j| zs_arr[[k, j]]));

        // Predict
        x = &f_k * &x;
        let mut p_pred = &f_k * &p * f_k.transpose() + &q_k;
        symmetrize_in_place(&mut p_pred);
        p = p_pred;

        // Linear update
        let h_t = h_k.transpose();
        let z_pred = &h_k * &x;
        let y = &z_k - &z_pred;
        let mut s = &h_k * &p * &h_t + &r_k;
        symmetrize_in_place(&mut s);
        let (si, _was_singular) = invert_innovation(&s)?;
        let k_gain = &p * &h_t * &si;

        x = &x + &k_gain * &y;
        let i_kh = &eye_x - &k_gain * &h_k;
        let mut p_new = &i_kh * &p * i_kh.transpose() + &k_gain * &r_k * k_gain.transpose();
        symmetrize_in_place(&mut p_new);
        p = p_new;

        // Log-likelihood via Cholesky log-det (same path as the per-step filter).
        let mahal2 = (y.transpose() * &si * &y)[(0, 0)].max(0.0);
        let ll = if let Some(chol) = s.clone().cholesky() {
            let l = chol.l();
            let mut log_det = 0.0;
            let mut ok = true;
            for i in 0..l.nrows() {
                let d = l[(i, i)];
                if d <= 0.0 || !d.is_finite() {
                    ok = false;
                    break;
                }
                log_det += d.ln();
            }
            log_det *= 2.0;
            if ok {
                -0.5 * ((dim_z as f64) * log_two_pi + log_det + mahal2)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            f64::NEG_INFINITY
        };
        ll_out[k] = ll;

        for j in 0..dim_x {
            xs_out[[k, j]] = x[j];
        }
        for r in 0..dim_x {
            for c in 0..dim_x {
                ps_out[[k, r, c]] = p[(r, c)];
            }
        }
    }

    Ok((
        xs_out.to_pyarray(py),
        ps_out.to_pyarray(py),
        ll_out.to_pyarray(py),
    ))
}

// ----------------------------------------------------------------------------
// Parallel batch step: many filters, one observation each
// ----------------------------------------------------------------------------
//
// Distinct from `batch_filter_linear` (one filter, many observations). Here
// we have a bank of M independent linear Kalman filters that share the same
// F/H/Q/R, each with its own (x, P), and each gets a single observation
// z_i. The M filter steps are executed in parallel via rayon. The GIL is
// released for the duration of the parallel section so worker threads don't
// serialize on Python state.
//
// Use cases: Monte-Carlo banks, particle filters, multi-target tracking
// where each target rides on the same dynamics + measurement model.
//
// Status code semantics (parallel-safe, no panics, no raise):
//   0 = ok, innovation covariance was PD (Cholesky succeeded)
//   1 = singular_innovation, fell back to pseudo-inverse for S
//   2 = failed, no inverse at all — state stays at the predict-step output
//       (no measurement update applied) and ll = -inf
//
// The linear path never touches `stable_cholesky` on P, so the jitter
// surface that the per-step path exposes via snapshot() does not apply.

/// Linear-step inversion helper that does not allocate via PyResult. Returns
/// (S^-1, status) where status mirrors the public contract of
/// batch_parallel_step. None means even the pseudo-inverse failed.
fn invert_innovation_noraise(s: &DMatrix<f64>) -> (Option<DMatrix<f64>>, i64) {
    if let Some(chol) = s.clone().cholesky() {
        return (Some(chol.inverse()), 0);
    }
    if let Some(inv) = s.clone().try_inverse() {
        return (Some(inv), 1);
    }
    if let Ok(pinv) = s.clone().svd(true, true).pseudo_inverse(1e-12) {
        return (Some(pinv), 1);
    }
    (None, 2)
}

/// Cholesky log-det of a (presumed) PD matrix. Returns None if a diagonal
/// entry is non-positive or non-finite.
fn cholesky_log_det(s: &DMatrix<f64>) -> Option<f64> {
    let chol = s.clone().cholesky()?;
    let l = chol.l();
    let mut log_det = 0.0;
    for i in 0..l.nrows() {
        let d = l[(i, i)];
        if d <= 0.0 || !d.is_finite() {
            return None;
        }
        log_det += d.ln();
    }
    Some(2.0 * log_det)
}

/// One predict + linear update step for a single (x, P) against observation
/// z. All shared matrices (F, H, Q, R, I) and constants are passed by
/// reference. Returns (x_new, P_new, log_likelihood, status). Allocations
/// are deliberately local so this is callable from within `par_iter`.
#[allow(clippy::too_many_arguments)]
fn linear_predict_update_step(
    x_in: &DVector<f64>,
    p_in: &DMatrix<f64>,
    z: &DVector<f64>,
    f: &DMatrix<f64>,
    f_t: &DMatrix<f64>,
    h: &DMatrix<f64>,
    h_t: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    eye_x: &DMatrix<f64>,
    log_two_pi: f64,
    dim_z: usize,
) -> (DVector<f64>, DMatrix<f64>, f64, i64) {
    // Predict
    let x_pred = f * x_in;
    let mut p_pred = f * p_in * f_t + q;
    symmetrize_in_place(&mut p_pred);

    // Innovation
    let z_pred = h * &x_pred;
    let y = z - &z_pred;
    let mut s = h * &p_pred * h_t + r;
    symmetrize_in_place(&mut s);

    let (si_opt, status) = invert_innovation_noraise(&s);
    let si = match si_opt {
        Some(m) => m,
        None => {
            // No update applied — return predict-step state.
            return (x_pred, p_pred, f64::NEG_INFINITY, status);
        }
    };

    let k_gain = &p_pred * h_t * &si;
    let x_new = &x_pred + &k_gain * &y;
    let i_kh = eye_x - &k_gain * h;
    let mut p_new = &i_kh * &p_pred * i_kh.transpose() + &k_gain * r * k_gain.transpose();
    symmetrize_in_place(&mut p_new);

    let mahal2 = (y.transpose() * &si * &y)[(0, 0)].max(0.0);
    let ll = if let Some(log_det) = cholesky_log_det(&s) {
        -0.5 * ((dim_z as f64) * log_two_pi + log_det + mahal2)
    } else {
        f64::NEG_INFINITY
    };

    (x_new, p_new, ll, status)
}

/// Parallel batch step: M independent linear KFs sharing F, H, Q, R; each
/// has its own (x_i, P_i) and a single observation z_i.
///
/// Inputs:
///   xs: (M, dim_x)
///   ps: (M, dim_x, dim_x)
///   zs: (M, dim_z)
///   f:  (dim_x, dim_x)
///   h:  (dim_z, dim_x)
///   q:  (dim_x, dim_x)
///   r:  (dim_z, dim_z)
///
/// Returns (xs_new, ps_new, log_likelihoods, status) with shapes
/// (M, dim_x), (M, dim_x, dim_x), (M,), (M,). Status code per filter:
///   0 = ok, 1 = singular_innovation fallback, 2 = no update applied.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn batch_parallel_step<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray2<f64>,
    ps: PyReadonlyArray3<f64>,
    zs: PyReadonlyArray2<f64>,
    f: PyReadonlyArray2<f64>,
    h: PyReadonlyArray2<f64>,
    q: PyReadonlyArray2<f64>,
    r: PyReadonlyArray2<f64>,
) -> PyResult<(
    &'py PyArray2<f64>,
    &'py PyArray3<f64>,
    &'py PyArray1<f64>,
    &'py PyArray1<i64>,
)> {
    let xs_arr = xs.as_array();
    let ps_arr = ps.as_array();
    let zs_arr = zs.as_array();
    let f_arr = f.as_array();
    let h_arr = h.as_array();
    let q_arr = q.as_array();
    let r_arr = r.as_array();

    let xs_shape = xs_arr.shape();
    if xs_shape.len() != 2 {
        return Err(PyValueError::new_err("xs must be 2D with shape (M, dim_x)"));
    }
    let m = xs_shape[0];
    let dim_x = xs_shape[1];
    if m == 0 {
        return Err(PyValueError::new_err("xs must contain at least one filter"));
    }
    if dim_x == 0 {
        return Err(PyValueError::new_err("dim_x must be positive"));
    }

    let zs_shape = zs_arr.shape();
    if zs_shape[0] != m {
        return Err(PyValueError::new_err(format!(
            "zs must have leading dimension {m}; got {}",
            zs_shape[0]
        )));
    }
    let dim_z = zs_shape[1];
    if dim_z == 0 {
        return Err(PyValueError::new_err("dim_z must be positive"));
    }

    let ps_shape = ps_arr.shape();
    if ps_shape != [m, dim_x, dim_x] {
        return Err(PyValueError::new_err(format!(
            "Ps must have shape ({m}, {dim_x}, {dim_x}); got ({}, {}, {})",
            ps_shape[0], ps_shape[1], ps_shape[2]
        )));
    }
    let check_2d = |name: &str, sh: &[usize], a: usize, b: usize| -> PyResult<()> {
        if sh[0] != a || sh[1] != b {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({a}, {b}); got ({}, {})",
                sh[0], sh[1]
            )));
        }
        Ok(())
    };
    check_2d("F", f_arr.shape(), dim_x, dim_x)?;
    check_2d("H", h_arr.shape(), dim_z, dim_x)?;
    check_2d("Q", q_arr.shape(), dim_x, dim_x)?;
    check_2d("R", r_arr.shape(), dim_z, dim_z)?;

    // Copy inputs out of Python-owned storage so we can drop the GIL.
    let f_mat = DMatrix::from_fn(dim_x, dim_x, |r, c| f_arr[[r, c]]);
    let h_mat = DMatrix::from_fn(dim_z, dim_x, |r, c| h_arr[[r, c]]);
    let q_mat = DMatrix::from_fn(dim_x, dim_x, |r, c| q_arr[[r, c]]);
    let r_mat = DMatrix::from_fn(dim_z, dim_z, |r, c| r_arr[[r, c]]);
    let f_t = f_mat.transpose();
    let h_t = h_mat.transpose();
    let eye_x = DMatrix::<f64>::identity(dim_x, dim_x);

    let mut xs_in: Vec<DVector<f64>> = Vec::with_capacity(m);
    let mut ps_in: Vec<DMatrix<f64>> = Vec::with_capacity(m);
    let mut zs_in: Vec<DVector<f64>> = Vec::with_capacity(m);
    for i in 0..m {
        xs_in.push(DVector::from_iterator(
            dim_x,
            (0..dim_x).map(|j| xs_arr[[i, j]]),
        ));
        ps_in.push(DMatrix::from_fn(dim_x, dim_x, |r, c| ps_arr[[i, r, c]]));
        zs_in.push(DVector::from_iterator(
            dim_z,
            (0..dim_z).map(|j| zs_arr[[i, j]]),
        ));
    }

    let log_two_pi = (2.0 * PI).ln();

    // Run the bank in parallel with the GIL released.
    let results: Vec<(DVector<f64>, DMatrix<f64>, f64, i64)> = py.allow_threads(|| {
        (0..m)
            .into_par_iter()
            .map(|i| {
                linear_predict_update_step(
                    &xs_in[i], &ps_in[i], &zs_in[i], &f_mat, &f_t, &h_mat, &h_t, &q_mat, &r_mat,
                    &eye_x, log_two_pi, dim_z,
                )
            })
            .collect()
    });

    let mut xs_out = numpy::ndarray::Array2::<f64>::zeros((m, dim_x));
    let mut ps_out = numpy::ndarray::Array3::<f64>::zeros((m, dim_x, dim_x));
    let mut ll_out = numpy::ndarray::Array1::<f64>::zeros(m);
    let mut status_out = numpy::ndarray::Array1::<i64>::zeros(m);

    for (i, (x_new, p_new, ll, status)) in results.into_iter().enumerate() {
        for j in 0..dim_x {
            xs_out[[i, j]] = x_new[j];
        }
        for r in 0..dim_x {
            for c in 0..dim_x {
                ps_out[[i, r, c]] = p_new[(r, c)];
            }
        }
        ll_out[i] = ll;
        status_out[i] = status;
    }

    Ok((
        xs_out.to_pyarray(py),
        ps_out.to_pyarray(py),
        ll_out.to_pyarray(py),
        status_out.to_pyarray(py),
    ))
}

// ----------------------------------------------------------------------------
// Square-root Cubature Kalman Filter (SR-CKF)
// ----------------------------------------------------------------------------
//
// Propagates the lower-triangular Cholesky factor `chol_p` (where
// `chol_p * chol_p^T = P`) directly. The standard CKF in this crate calls
// `stable_cholesky(P)` inside every `predict`/`update`, silently adding jitter
// to the diagonal whenever P drifts toward non-positive-definite. The
// square-root form never re-decomposes P during the filter loop: predict and
// update both produce the next factor via QR + rank-1 Cholesky downdates, so
// the "silent jitter" hazard at predict-time goes away entirely.
//
// References:
//   Arasaratnam & Haykin, "Cubature Kalman Filters", IEEE TAC 54(6), 2009 —
//   square-root variant (Algorithm 2).
//   Van der Merwe (PhD thesis, 2004) — SR-UKF Algorithm 3.2; the CKF case is
//   the symmetric-weight specialisation (no separate central-point update).

#[pyclass]
pub struct SquareRootCubatureKalmanFilter {
    dim_x: usize,
    dim_z: usize,
    dt: f64,
    x: DVector<f64>,
    // Lower-triangular state-cov factor: chol_p * chol_p^T = P.
    chol_p: DMatrix<f64>,
    // Lower-triangular process-noise factor: chol_q * chol_q^T = Q.
    chol_q: DMatrix<f64>,
    // Lower-triangular measurement-noise factor: chol_r * chol_r^T = R.
    chol_r: DMatrix<f64>,
    // Innovation factor from the last update: s_innov * s_innov^T = S.
    s_innov: DMatrix<f64>,
    k: DMatrix<f64>,
    y: DVector<f64>,
    z: DVector<f64>,
    x_prior: DVector<f64>,
    chol_p_prior: DMatrix<f64>,
    x_post: DVector<f64>,
    chol_p_post: DMatrix<f64>,
    z_pred: DVector<f64>,
    log_likelihood: f64,
    likelihood: f64,
    mahalanobis: f64,
    nis: f64,
    // SR-CKF diagnostics. `jitter_count` is kept for API symmetry with the
    // standard CKF, but the square-root path only invokes `stable_cholesky`
    // when the user (re)seeds P / Q / R via `set_state` — never inside the
    // filter loop. `downdate_fallback_count` increments when a rank-1
    // Cholesky downdate of the posterior factor would produce a non-PD
    // result and we fall back to a fresh Cholesky of the rebuilt P_post.
    last_jitter: f64,
    max_jitter: f64,
    jitter_count: u64,
    singular_innovation_count: u64,
    downdate_fallback_count: u64,
}

#[pymethods]
impl SquareRootCubatureKalmanFilter {
    #[new]
    fn new(dim_x: usize, dim_z: usize, dt: f64) -> PyResult<Self> {
        if dim_x == 0 || dim_z == 0 {
            return Err(PyValueError::new_err("dim_x and dim_z must be positive"));
        }
        if !dt.is_finite() {
            return Err(PyValueError::new_err("dt must be finite"));
        }
        Ok(Self {
            dim_x,
            dim_z,
            dt,
            x: DVector::zeros(dim_x),
            chol_p: DMatrix::identity(dim_x, dim_x),
            chol_q: DMatrix::identity(dim_x, dim_x),
            chol_r: DMatrix::identity(dim_z, dim_z),
            s_innov: DMatrix::identity(dim_z, dim_z),
            k: DMatrix::zeros(dim_x, dim_z),
            y: DVector::zeros(dim_z),
            z: DVector::zeros(dim_z),
            x_prior: DVector::zeros(dim_x),
            chol_p_prior: DMatrix::identity(dim_x, dim_x),
            x_post: DVector::zeros(dim_x),
            chol_p_post: DMatrix::identity(dim_x, dim_x),
            z_pred: DVector::zeros(dim_z),
            log_likelihood: f64::NAN,
            likelihood: f64::NAN,
            mahalanobis: f64::NAN,
            nis: f64::NAN,
            last_jitter: 0.0,
            max_jitter: 0.0,
            jitter_count: 0,
            singular_innovation_count: 0,
            downdate_fallback_count: 0,
        })
    }

    /// Seed the filter from a covariance-space description.
    ///
    /// P, Q, R are accepted as full covariance matrices; their Cholesky
    /// factors are computed once here via `stable_cholesky`. Any jitter that
    /// had to be added at seeding time is recorded in the counters, but the
    /// subsequent filter loop never touches `stable_cholesky` on P again —
    /// so the typical predict-time jitter accumulation seen in the standard
    /// CKF cannot happen.
    #[pyo3(signature = (x, p, q, r))]
    fn set_state(
        &mut self,
        x: PyReadonlyArray1<f64>,
        p: PyReadonlyArray2<f64>,
        q: PyReadonlyArray2<f64>,
        r: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.x = pyarray1_to_dvector(x, self.dim_x, "x")?;
        let p_mat = pyarray2_to_dmatrix(p, self.dim_x, self.dim_x, "P")?;
        let q_mat = pyarray2_to_dmatrix(q, self.dim_x, self.dim_x, "Q")?;
        let r_mat = pyarray2_to_dmatrix(r, self.dim_z, self.dim_z, "R")?;
        let (chol_p, jitter_p) = stable_cholesky(&p_mat)?;
        let (chol_q, jitter_q) = stable_cholesky(&q_mat)?;
        let (chol_r, jitter_r) = stable_cholesky(&r_mat)?;
        self.chol_p = chol_p;
        self.chol_q = chol_q;
        self.chol_r = chol_r;
        // Surface the worst seed-time jitter so callers can see if their
        // P/Q/R were degenerate. We do NOT bump jitter_count for these —
        // they're a one-shot cost at seeding, not the per-step silent
        // jitter the standard CKF accumulates.
        let worst = jitter_p.max(jitter_q).max(jitter_r);
        if worst > self.last_jitter {
            self.last_jitter = worst;
        }
        if worst > self.max_jitter {
            self.max_jitter = worst;
        }
        Ok(())
    }

    #[pyo3(signature = (fx, dt=None, fx_args=None))]
    fn predict_custom(
        &mut self,
        py: Python<'_>,
        fx: PyObject,
        dt: Option<f64>,
        fx_args: Option<&PyTuple>,
    ) -> PyResult<()> {
        let local_dt = dt.unwrap_or(self.dt);
        if !local_dt.is_finite() {
            return Err(PyValueError::new_err("dt must be finite"));
        }
        let n = self.dim_x;
        let sigma = cubature_points_from_factor(&self.x, &self.chol_p);
        let args = fx_args.unwrap_or_else(|| PyTuple::empty(py));
        let propagated = call_model_vectorized(py, &fx, &sigma, Some(local_dt), args, n)?;

        let mean = row_mean(&propagated, n);
        let w_sqrt = 1.0 / ((2 * n) as f64).sqrt();

        // Stacked QR input M of shape (3n × n):
        //   rows 0..2n : w_sqrt * (propagated_row_i - mean)
        //   rows 2n..3n: rows of chol_q^T (i.e., columns of chol_q)
        // M^T · M = empirical_cov + chol_q · chol_q^T = P_pred.
        let mut m = DMatrix::<f64>::zeros(3 * n, n);
        for i in 0..(2 * n) {
            for j in 0..n {
                m[(i, j)] = w_sqrt * (propagated[(i, j)] - mean[j]);
            }
        }
        for row in 0..n {
            for col in 0..n {
                // Row `row` of chol_q^T == column `row` of chol_q == chol_q[(col, row)].
                m[(2 * n + row, col)] = self.chol_q[(col, row)];
            }
        }

        let chol_p_new = qr_to_lower_factor(m, n)?;
        self.x = mean;
        self.chol_p = chol_p_new;
        self.x_prior = self.x.clone();
        self.chol_p_prior = self.chol_p.clone();
        Ok(())
    }

    /// Clear cached innovation / likelihood diagnostics. Used by the Python
    /// wrapper when a measurement is skipped (`update(z=None)`), so callers
    /// don't accidentally re-read stale values.
    fn clear_update_diagnostics(&mut self) {
        self.y = DVector::zeros(self.dim_z);
        self.z = DVector::zeros(self.dim_z);
        self.z_pred = DVector::zeros(self.dim_z);
        self.s_innov = DMatrix::identity(self.dim_z, self.dim_z);
        self.k = DMatrix::zeros(self.dim_x, self.dim_z);
        self.log_likelihood = f64::NAN;
        self.likelihood = f64::NAN;
        self.mahalanobis = f64::NAN;
        self.nis = f64::NAN;
    }

    fn reset_jitter_counters(&mut self) {
        self.last_jitter = 0.0;
        self.max_jitter = 0.0;
        self.jitter_count = 0;
        self.singular_innovation_count = 0;
        self.downdate_fallback_count = 0;
    }

    #[pyo3(signature = (z, hx, r=None, hx_args=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        hx: PyObject,
        r: Option<PyReadonlyArray2<f64>>,
        hx_args: Option<&PyTuple>,
    ) -> PyResult<()> {
        let n = self.dim_x;
        let m_dim = self.dim_z;
        let z_vec = pyarray1_to_dvector(z, m_dim, "z")?;
        let chol_r_local = if let Some(mat) = r {
            let r_mat = pyarray2_to_dmatrix(mat, m_dim, m_dim, "R")?;
            let (chol_r, jitter) = stable_cholesky(&r_mat)?;
            if jitter > 0.0 {
                self.last_jitter = jitter;
                if jitter > self.max_jitter {
                    self.max_jitter = jitter;
                }
            }
            chol_r
        } else {
            self.chol_r.clone()
        };

        let sigma = cubature_points_from_factor(&self.x, &self.chol_p);
        let args = hx_args.unwrap_or_else(|| PyTuple::empty(py));
        let z_sigma = call_model_vectorized(py, &hx, &sigma, None, args, m_dim)?;

        let z_pred = row_mean(&z_sigma, m_dim);
        let w_sqrt = 1.0 / ((2 * n) as f64).sqrt();

        // Centred and weight-scaled sigma-point matrices.
        let mut x_centered = DMatrix::<f64>::zeros(2 * n, n);
        let mut z_centered = DMatrix::<f64>::zeros(2 * n, m_dim);
        for i in 0..(2 * n) {
            for j in 0..n {
                x_centered[(i, j)] = w_sqrt * (sigma[(i, j)] - self.x[j]);
            }
            for j in 0..m_dim {
                z_centered[(i, j)] = w_sqrt * (z_sigma[(i, j)] - z_pred[j]);
            }
        }

        // QR of stacked [z_centered; chol_r^T] gives the innovation factor.
        let mut m_zz = DMatrix::<f64>::zeros(2 * n + m_dim, m_dim);
        for i in 0..(2 * n) {
            for j in 0..m_dim {
                m_zz[(i, j)] = z_centered[(i, j)];
            }
        }
        for row in 0..m_dim {
            for col in 0..m_dim {
                m_zz[(2 * n + row, col)] = chol_r_local[(col, row)];
            }
        }
        let s_innov = qr_to_lower_factor(m_zz, m_dim)?;

        // Cross-cov: P_xz = X_centered^T · Z_centered  (n × m).
        let p_xz = x_centered.tr_mul(&z_centered);

        // Gain: K = P_xz · (S_innov · S_innov^T)^{-1}
        //       K^T = (S_innov^T)^{-1} · S_innov^{-1} · P_xz^T
        // Solve K^T in two triangular steps so we never form the inverse.
        let mut kt = p_xz.transpose(); // (m × n)
        if !s_innov.solve_lower_triangular_mut(&mut kt) {
            self.singular_innovation_count = self.singular_innovation_count.saturating_add(1);
            return Err(PyRuntimeError::new_err(
                "SR-CKF innovation factor singular during gain solve",
            ));
        }
        if !s_innov.tr_solve_lower_triangular_mut(&mut kt) {
            self.singular_innovation_count = self.singular_innovation_count.saturating_add(1);
            return Err(PyRuntimeError::new_err(
                "SR-CKF innovation factor singular during gain solve",
            ));
        }
        let k = kt.transpose(); // (n × m)

        let y = &z_vec - &z_pred;
        self.x = &self.x + &k * &y;

        // Posterior factor via m sequential rank-1 Cholesky downdates:
        //   chol_p_post · chol_p_post^T = chol_p · chol_p^T - U · U^T,
        // where U = K · S_innov (n × m). If a downdate would break PD we
        // fall back to rebuilding P_post explicitly and Cholesky-ing it.
        let u = &k * &s_innov;
        let mut chol_p_new = self.chol_p.clone();
        let mut downdate_ok = true;
        for col_idx in 0..m_dim {
            let mut u_col = u.column(col_idx).clone_owned();
            if cholesky_downdate(&mut chol_p_new, &mut u_col).is_err() {
                downdate_ok = false;
                break;
            }
        }
        if downdate_ok {
            self.chol_p = chol_p_new;
        } else {
            let p_old = &self.chol_p * self.chol_p.transpose();
            let mut p_post = p_old - &u * u.transpose();
            symmetrize_in_place(&mut p_post);
            let (chol, jitter) = stable_cholesky(&p_post)?;
            self.chol_p = chol;
            if jitter > 0.0 {
                self.last_jitter = jitter;
                if jitter > self.max_jitter {
                    self.max_jitter = jitter;
                }
                self.jitter_count = self.jitter_count.saturating_add(1);
            }
            self.downdate_fallback_count = self.downdate_fallback_count.saturating_add(1);
        }

        // Diagnostics. Mahalanobis² = ||S_innov^{-1} · y||².
        let mut v = y.clone();
        let mahal2 = if s_innov.solve_lower_triangular_mut(&mut v) {
            v.dot(&v).max(0.0)
        } else {
            f64::NAN
        };
        let mut log_det = 0.0;
        let mut log_det_ok = true;
        for i in 0..m_dim {
            let d = s_innov[(i, i)];
            if !(d.is_finite() && d > 0.0) {
                log_det_ok = false;
                break;
            }
            log_det += d.ln();
        }
        log_det *= 2.0;
        if log_det_ok && mahal2.is_finite() {
            self.log_likelihood = -0.5 * ((m_dim as f64) * (2.0 * PI).ln() + log_det + mahal2);
            self.likelihood = self.log_likelihood.exp();
            self.nis = mahal2;
            self.mahalanobis = mahal2.sqrt();
        } else {
            self.log_likelihood = f64::NEG_INFINITY;
            self.likelihood = 0.0;
            self.nis = f64::NAN;
            self.mahalanobis = f64::NAN;
        }

        self.z = z_vec;
        self.z_pred = z_pred;
        self.k = k;
        self.y = y;
        self.s_innov = s_innov;
        self.x_post = self.x.clone();
        self.chol_p_post = self.chol_p.clone();
        Ok(())
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        let p = &self.chol_p * self.chol_p.transpose();
        let p_prior = &self.chol_p_prior * self.chol_p_prior.transpose();
        let p_post = &self.chol_p_post * self.chol_p_post.transpose();
        let q = &self.chol_q * self.chol_q.transpose();
        let r = &self.chol_r * self.chol_r.transpose();
        let s = &self.s_innov * self.s_innov.transpose();
        out.set_item("x", self.x.as_slice().to_pyarray(py))?;
        out.set_item("chol_P", dmatrix_to_pyarray(py, &self.chol_p)?)?;
        out.set_item("P", dmatrix_to_pyarray(py, &p)?)?;
        out.set_item("chol_Q", dmatrix_to_pyarray(py, &self.chol_q)?)?;
        out.set_item("Q", dmatrix_to_pyarray(py, &q)?)?;
        out.set_item("chol_R", dmatrix_to_pyarray(py, &self.chol_r)?)?;
        out.set_item("R", dmatrix_to_pyarray(py, &r)?)?;
        out.set_item("K", dmatrix_to_pyarray(py, &self.k)?)?;
        out.set_item("y", self.y.as_slice().to_pyarray(py))?;
        out.set_item("z", self.z.as_slice().to_pyarray(py))?;
        out.set_item("S", dmatrix_to_pyarray(py, &s)?)?;
        out.set_item("S_innov", dmatrix_to_pyarray(py, &self.s_innov)?)?;
        out.set_item("x_prior", self.x_prior.as_slice().to_pyarray(py))?;
        out.set_item("P_prior", dmatrix_to_pyarray(py, &p_prior)?)?;
        out.set_item("x_post", self.x_post.as_slice().to_pyarray(py))?;
        out.set_item("P_post", dmatrix_to_pyarray(py, &p_post)?)?;
        out.set_item("z_pred", self.z_pred.as_slice().to_pyarray(py))?;
        out.set_item("log_likelihood", self.log_likelihood)?;
        out.set_item("likelihood", self.likelihood)?;
        out.set_item("mahalanobis", self.mahalanobis)?;
        out.set_item("nis", self.nis)?;
        out.set_item("last_jitter", self.last_jitter)?;
        out.set_item("max_jitter", self.max_jitter)?;
        out.set_item("jitter_count", self.jitter_count)?;
        out.set_item("singular_innovation_count", self.singular_innovation_count)?;
        out.set_item("downdate_fallback_count", self.downdate_fallback_count)?;
        Ok(out.to_object(py))
    }
}

#[inline]
fn cubature_points_from_factor(x: &DVector<f64>, chol_p: &DMatrix<f64>) -> DMatrix<f64> {
    let n = x.nrows();
    let scale = (n as f64).sqrt();
    let mut sigma = DMatrix::<f64>::zeros(2 * n, n);
    for k in 0..n {
        for j in 0..n {
            let offset = scale * chol_p[(j, k)];
            sigma[(k, j)] = x[j] + offset;
            sigma[(n + k, j)] = x[j] - offset;
        }
    }
    sigma
}

/// Given M of shape (k, n) with k >= n, return a lower-triangular S (n × n)
/// satisfying `S · S^T = M^T · M`. Computed via QR; the diagonal of S is
/// normalised non-negative for a canonical factor.
fn qr_to_lower_factor(m: DMatrix<f64>, n: usize) -> PyResult<DMatrix<f64>> {
    if m.nrows() < n {
        return Err(PyRuntimeError::new_err(
            "qr_to_lower_factor requires at least n rows",
        ));
    }
    let qr = m.qr();
    let r_full = qr.r();
    let mut s = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            s[(i, j)] = r_full[(j, i)];
        }
    }
    for j in 0..n {
        if s[(j, j)] < 0.0 {
            for i in j..n {
                s[(i, j)] = -s[(i, j)];
            }
        }
    }
    Ok(s)
}

/// Cholesky rank-1 downdate via hyperbolic Givens-style rotations.
///
/// Modifies the lower-triangular `l` in place such that the new factor
/// satisfies `l_new · l_new^T = l · l^T - x · x^T`. `x` is consumed (mutated)
/// to drive the rotation sequence; callers should clone the column they pass
/// in if they need the original later.
///
/// Returns Err if the downdate would produce a non-PD matrix (i.e., x has
/// non-trivial energy in the column space of `l`, so the result wouldn't be
/// a real Cholesky factor). Callers should fall back to rebuilding P and
/// taking a fresh Cholesky in that case.
fn cholesky_downdate(l: &mut DMatrix<f64>, x: &mut DVector<f64>) -> Result<(), ()> {
    let n = l.nrows();
    for i in 0..n {
        let lii = l[(i, i)];
        let xi = x[i];
        let r_sq = lii * lii - xi * xi;
        if !r_sq.is_finite() || r_sq <= 0.0 {
            return Err(());
        }
        let r = r_sq.sqrt();
        let c = lii / r;
        let s = xi / r;
        l[(i, i)] = r;
        for j in (i + 1)..n {
            let lji = l[(j, i)];
            let xj = x[j];
            l[(j, i)] = c * lji - s * xj;
            x[j] = -s * lji + c * xj;
        }
    }
    Ok(())
}

#[pymodule]
fn _rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CubatureKalmanFilter>()?;
    m.add_class::<SquareRootCubatureKalmanFilter>()?;
    m.add_function(wrap_pyfunction!(rts_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(batch_filter_linear, m)?)?;
    m.add_function(wrap_pyfunction!(batch_parallel_step, m)?)?;
    Ok(())
}

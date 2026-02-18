use nalgebra::{DMatrix, DVector};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
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
}

#[pymethods]
impl CubatureKalmanFilter {
    #[new]
    fn new(dim_x: usize, dim_z: usize, dt: f64) -> PyResult<Self> {
        if dim_x == 0 || dim_z == 0 {
            return Err(PyValueError::new_err("dim_x and dim_z must be positive"));
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
        let sigma = cubature_points(&self.x, &self.p)?;
        let args = fx_args.unwrap_or_else(|| PyTuple::empty(py));
        let propagated = call_model_vectorized(py, &fx, &sigma, Some(local_dt), args, self.dim_x)?;

        let m = propagated.nrows();
        let w = 1.0 / (m as f64);

        let mean = row_mean(&propagated, self.dim_x);
        let second_moment = propagated.transpose() * &propagated * w;
        let cov = symmetrize(&(second_moment - (&mean * mean.transpose()) + &self.q));

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

        let sigma = cubature_points(&self.x, &self.p)?;
        let args = hx_args.unwrap_or_else(|| PyTuple::empty(py));
        let z_sigma = call_model_vectorized(py, &hx, &sigma, None, args, self.dim_z)?;

        let m = sigma.nrows();
        let w = 1.0 / (m as f64);

        let z_pred = row_mean(&z_sigma, self.dim_z);
        let pxz = (sigma.transpose() * &z_sigma) * w - (&self.x * z_pred.transpose());
        let pzz = symmetrize(&((z_sigma.transpose() * &z_sigma) * w - (&z_pred * z_pred.transpose()) + &r_mat));

        let si = if let Some(inv) = pzz.clone().try_inverse() {
            inv
        } else {
            pzz.clone()
                .svd(true, true)
                .pseudo_inverse(1e-12)
                .map_err(|_| PyRuntimeError::new_err("failed to invert innovation covariance"))?
        };

        let k = &pxz * &si;
        let y = &z_vec - &z_pred;

        self.x = &self.x + &k * &y;
        self.p = symmetrize(&(&self.p - &k * &pzz * k.transpose()));

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
        let z_vec = pyarray1_to_dvector(z, self.dim_z, "z")?;
        let (m_n, m_d) = magnetic_reference_terms(&z_vec)?;

        let sigma = cubature_points(&self.x, &self.p)?;
        let z_sigma = paper_observation_model(&sigma, m_n, m_d);

        let mut r_mat = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            r_mat[(i, i)] = sigma_acc2;
            r_mat[(i + 3, i + 3)] = sigma_mag2;
        }

        let m = sigma.nrows();
        let w = 1.0 / (m as f64);

        let z_pred = row_mean(&z_sigma, self.dim_z);
        let pxz = (sigma.transpose() * &z_sigma) * w - (&self.x * z_pred.transpose());
        let pzz = symmetrize(&((z_sigma.transpose() * &z_sigma) * w - (&z_pred * z_pred.transpose()) + &r_mat));

        let si = if let Some(inv) = pzz.clone().try_inverse() {
            inv
        } else {
            pzz.clone()
                .svd(true, true)
                .pseudo_inverse(1e-12)
                .map_err(|_| PyRuntimeError::new_err("failed to invert innovation covariance"))?
        };

        let k = &pxz * &si;
        let y = &z_vec - &z_pred;

        self.x = &self.x + &k * &y;
        self.p = symmetrize(&(&self.p - &k * &pzz * k.transpose()));

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
            return Err(PyValueError::new_err("quaternion norm must be finite and positive"));
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
        Ok(out.to_object(py))
    }
}

impl CubatureKalmanFilter {
    fn predict_kckf_linear(&mut self, f: &DMatrix<f64>) {
        self.x = f * &self.x;
        self.p = symmetrize(&(f * &self.p * f.transpose() + &self.q));
        self.x_prior = self.x.clone();
        self.p_prior = self.p.clone();
    }

    fn predict_ckf_linear(&mut self, f: &DMatrix<f64>) -> PyResult<()> {
        let sigma = cubature_points(&self.x, &self.p)?;
        // Propagate each row-vector cubature point through the linear model.
        let propagated = sigma * f.transpose();
        let m = propagated.nrows();
        let w = 1.0 / (m as f64);

        let mean = row_mean(&propagated, self.dim_x);
        let second_moment = propagated.transpose() * &propagated * w;
        let cov = symmetrize(&(second_moment - (&mean * mean.transpose()) + &self.q));

        self.x = mean;
        self.p = cov;
        self.x_prior = self.x.clone();
        self.p_prior = self.p.clone();
        Ok(())
    }

    fn update_likelihood_terms(&mut self) {
        let det = self.s.clone().lu().determinant();
        if !det.is_finite() || det <= 0.0 {
            self.log_likelihood = f64::NEG_INFINITY;
            self.likelihood = 0.0;
            self.mahalanobis = f64::INFINITY;
            return;
        }
        let mahal2 = (self.y.transpose() * &self.si * &self.y)[(0, 0)];
        self.mahalanobis = mahal2.max(0.0).sqrt();
        self.log_likelihood = -0.5 * ((self.dim_z as f64) * (2.0 * PI).ln() + det.ln() + mahal2);
        self.likelihood = self.log_likelihood.exp();
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
            if dim_x % 2 != 0 {
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
            if dim_x % 3 != 0 {
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
        _ => Err(PyValueError::new_err(format!("unsupported model_type: {model_type}"))),
    }
}

fn cubature_points(x: &DVector<f64>, p: &DMatrix<f64>) -> PyResult<DMatrix<f64>> {
    let n = x.nrows();
    let chol = stable_cholesky(p)?;
    let scale = (n as f64).sqrt();

    let mut sigma = DMatrix::<f64>::zeros(2 * n, n);
    for k in 0..n {
        for j in 0..n {
            // Match FilterPy's scipy-based upper-triangular convention.
            let offset = scale * chol[(j, k)];
            sigma[(k, j)] = x[j] + offset;
            sigma[(n + k, j)] = x[j] - offset;
        }
    }
    Ok(sigma)
}

fn stable_cholesky(p: &DMatrix<f64>) -> PyResult<DMatrix<f64>> {
    let n = p.nrows();
    let eye = DMatrix::<f64>::identity(n, n);
    let mut jitter = 0.0_f64;

    for _ in 0..8 {
        let candidate = p + eye.scale(jitter);
        if let Some(chol) = candidate.cholesky() {
            return Ok(chol.l());
        }
        jitter = if jitter == 0.0 { 1e-12 } else { jitter * 10.0 };
    }
    Err(PyRuntimeError::new_err("unable to compute stable Cholesky factor"))
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
        out[(i, 3)] = (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * m_n + 2.0 * (q1 * q3 - q0 * q2) * m_d;
        out[(i, 4)] = 2.0 * (q1 * q2 - q0 * q3) * m_n + 2.0 * (q2 * q3 + q0 * q1) * m_d;
        out[(i, 5)] = 2.0 * (q1 * q3 + q0 * q2) * m_n + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * m_d;
    }
    out
}

fn row_mean(values: &DMatrix<f64>, dim: usize) -> DVector<f64> {
    let mut mean = DVector::<f64>::zeros(dim);
    let w = 1.0 / (values.nrows() as f64);
    for i in 0..values.nrows() {
        for j in 0..dim {
            mean[j] += w * values[(i, j)];
        }
    }
    mean
}

fn symmetrize(mat: &DMatrix<f64>) -> DMatrix<f64> {
    (mat + mat.transpose()) * 0.5
}

fn pyarray1_to_dvector(arr: PyReadonlyArray1<f64>, expected: usize, name: &str) -> PyResult<DVector<f64>> {
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
    let mut out = DMatrix::<f64>::zeros(expected_rows, expected_cols);
    for i in 0..expected_rows {
        for j in 0..expected_cols {
            out[(i, j)] = data[(i, j)];
        }
    }
    Ok(out)
}

fn dmatrix_to_pyarray<'py>(py: Python<'py>, mat: &DMatrix<f64>) -> PyResult<&'py PyArray2<f64>> {
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(mat.nrows());
    for i in 0..mat.nrows() {
        let mut row: Vec<f64> = Vec::with_capacity(mat.ncols());
        for j in 0..mat.ncols() {
            row.push(mat[(i, j)]);
        }
        rows.push(row);
    }
    PyArray2::from_vec2(py, &rows).map_err(|_| PyRuntimeError::new_err("failed to create numpy matrix"))
}

#[pymodule]
fn _rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CubatureKalmanFilter>()?;
    Ok(())
}

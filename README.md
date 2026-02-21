# turbo-ckf

`turbo-ckf` is a Rust-backed Cubature Kalman Filter package for high-throughput prediction/update loops. Implemented here purely as an experiment after reading the paper.

## What This Package Optimizes

- Fast prediction with built-in linear models:
  - `predict_standard_model(...)`
  - `predict_standard_model_ckf(...)`
  - `predict_linear_model(F)`
  - `predict_linear_model_ckf(F)`
- Fast AHRS update path:
  - `update_paper_ahrs(...)`

`predict(...)` and `update(...)` also run through Rust, but callback cost in Python can dominate if your models are heavy.

## Callback Contract

Custom `fx` and `hx` must be vectorized:

- Input shape is `(2 * dim_x, dim_x)`.
- `fx` output shape must be `(2 * dim_x, dim_x)`.
- `hx` output shape must be `(2 * dim_x, dim_z)`.

If you pass pointwise callbacks, `TurboCKF` raises immediately.

## Install (Local Dev)

From `turbo-ckf/`:

```bash
bash turbo_ckf/setup_env.sh
```

## Usage

```python
from turbo_ckf import TurboCKF

kf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_vectorized, fx=fx_vectorized)
kf.predict_standard_model("constant_velocity")
kf.update(z)
```

AHRS path:

```python
kf.predict_linear_model(Fk)
kf.update_paper_ahrs(z6, sigma_acc2=1e-2, sigma_mag2=1e-2)
```

## Benchmarks

### Paper-Reported Targets (Shing et al., arXiv:2602.12283)

These are the KCKF-vs-CKF results reported in the paper:

- MacBook Pro 2021 (M1 Pro): KCKF `0.110 ms` vs CKF `0.135 ms` (`18.79%` lower time, about `1.23x` faster).
- Raspberry Pi 4 Model B: KCKF `1.28 ms` vs CKF `1.51 ms` (`15.15%` lower time, about `1.18x` faster).

### Local Results In This Repo

Measured on **February 21, 2026** on Apple M4 Max, Python 3.12.0, NumPy 2.4.2.

From:

```bash
.venv-turbo-ckf/bin/python turbo_ckf_tests/verify_before_after.py --steps 50000 --repeats 7 --warmup 1 --parity-steps 500
```

Median runtime per 50k predict+update steps:

- FilterPy (`filterpy_predict_update`): `2.354375 s` (baseline)
- TurboCKF callback path (`turbo_callback_predict_update`): `0.662003 s` (`3.56x` vs FilterPy)
- TurboCKF standard model KCKF (`turbo_standard_model_predict_update`): `0.539281 s` (`4.37x` vs FilterPy)
- TurboCKF standard model CKF (`turbo_standard_model_ckf_predict_update`): `0.549905 s` (`4.28x` vs FilterPy)
- KCKF-vs-CKF full-step speedup in this setup: `1.02x` (`0.549905 / 0.539281`)

From:

```bash
.venv-turbo-ckf/bin/python turbo_ckf_tests/benchmark_paper.py
```

- KCKF-vs-CKF (prediction only, backend): `1.67x`
- KCKF-vs-CKF (full step, backend): `1.09x`
- KCKF-vs-CKF (full step, wrapper): `1.03x`

Interpretation:

- The largest gains here are from the Rust-backed implementation versus pure-Python FilterPy loops (`~3.5x` to `~4.4x` in these runs).
- The direct KCKF-vs-CKF algorithmic gain is present but smaller on full-step end-to-end runs, and larger when isolating prediction equations.

## Research Basis

- This repo is an implementation of the KCKF AHRS equations described in:
  - Shing, Y. C., et al., "KCKF: A Fast and Stable Quaternion-Based Orientation Estimator", arXiv:2602.12283 (2026), https://arxiv.org/abs/2602.12283.
- Credit for the method belongs to the paper authors.

## License

Dual-licensed under either:
- MIT (`LICENSE-MIT`)
- Apache-2.0 (`LICENSE-APACHE`)

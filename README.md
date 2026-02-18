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

## Research Basis

- This repo is an implementation of the KCKF AHRS equations described in:
  - Shing, Y. C., et al., "KCKF: A Fast and Stable Quaternion-Based Orientation Estimator", arXiv:2602.12283 (2026), https://arxiv.org/abs/2602.12283.
- Credit for the method belongs to the paper authors.

## License

Dual-licensed under either:
- MIT (`LICENSE-MIT`)
- Apache-2.0 (`LICENSE-APACHE`)

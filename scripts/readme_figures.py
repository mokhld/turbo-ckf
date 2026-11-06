#!/usr/bin/env python3
"""Draw the figures used in README.md.

Writes a light and a dark SVG of each figure to docs/img/:

- tracking: a constant-velocity tracker following noisy position readings
- cubature: how the cubature points are built and pushed through a nonlinear model
- benchmark: time per step against FilterPy, read from a verify_before_after.py report

Every filter number in the figures comes from turbo_ckf itself: the cubature points
are the ones Rust hands to ``fx``, and the nonlinear moments are a real ``predict()``.

Run from the repository root after building the extension::

    .venv-turbo-ckf/bin/python turbo_ckf_tests/verify_before_after.py \\
        --steps 50000 --repeats 7 --warmup 1 --parity-steps 500 \\
        --output docs/img/benchmark.json
    .venv-turbo-ckf/bin/python scripts/readme_figures.py --machine "Apple M4 Max"

Needs matplotlib (the ``docs`` extra, or ``pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from turbo_ckf import TurboCKF  # noqa: E402

THEMES = {
    "light": {
        "fg": "#1f2328",
        "muted": "#59636e",
        "faint": "#d1d9e0",
        "accent": "#bc4c00",
        "accent2": "#0969da",
    },
    "dark": {
        "fg": "#e6edf3",
        "muted": "#9198a1",
        "faint": "#3d444d",
        "accent": "#f0883e",
        "accent2": "#4493f8",
    },
}

BENCH_LABELS = {
    "filterpy_predict_update": "FilterPy CubatureKalmanFilter\npredict + update",
    "turbo_callback_predict_update": "TurboCKF predict + update\n(Python fx and hx callbacks)",
    "turbo_standard_model_predict_update": "TurboCKF built-in model (KCKF)\n+ update",
    "turbo_standard_model_ckf_predict_update": "TurboCKF built-in model (CKF)\n+ update",
}


# ---------------------------------------------------------------------------
# Models (vectorized: rows are cubature points)
# ---------------------------------------------------------------------------


def fx_constant_velocity(sigmas, dt):
    out = sigmas.copy()
    out[:, 0] += dt * sigmas[:, 1]
    return out


def hx_position(sigmas):
    return sigmas[:, :1]


def fx_polar_to_cartesian(sigmas, dt):
    out = np.empty_like(sigmas)
    out[:, 0] = sigmas[:, 0] * np.cos(sigmas[:, 1])
    out[:, 1] = sigmas[:, 0] * np.sin(sigmas[:, 1])
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def tracking_data():
    """Constant-velocity tracker on noisy position readings (seeded)."""
    rng = np.random.default_rng(42)
    dt, n, sigma_z = 0.1, 300, 0.5
    vel = 1.5 + np.cumsum(rng.normal(0.0, 0.05, n))  # slow random walk
    pos = np.cumsum(vel * dt)
    zs = pos + rng.normal(0.0, sigma_z, n)

    kf = TurboCKF(dim_x=2, dim_z=1, dt=dt, hx=hx_position, fx=fx_constant_velocity)
    kf.x = [0.0, 0.0]
    kf.P = [10.0, 10.0]
    q = 0.05**2
    kf.Q = [[q * dt**2 / 3, q * dt / 2], [q * dt / 2, q]]
    kf.R = sigma_z**2
    result = kf.run(zs)
    t = dt * np.arange(1, n + 1)
    return t, pos, vel, zs, result.xs, result.Ps


def cubature_points_from_filter(x, P):
    """Return the points turbo_ckf's Rust predict passes to fx."""
    captured = {}

    def spy(sigmas, dt):
        captured["sigmas"] = sigmas.copy()
        return sigmas

    kf = TurboCKF(dim_x=2, dim_z=1, dt=1.0, hx=hx_position, fx=spy)
    kf.x = x
    kf.P = P
    kf.Q = 0.0
    kf.predict()
    return captured["sigmas"]


def nonlinear_moments(mean, P):
    """Cubature mean/covariance of polar -> Cartesian, plus the mapped points."""
    captured = {}

    def fx(sigmas, dt):
        out = fx_polar_to_cartesian(sigmas, dt)
        captured["mapped"] = out.copy()
        return out

    kf = TurboCKF(dim_x=2, dim_z=1, dt=1.0, hx=hx_position, fx=fx)
    kf.x = mean
    kf.P = P
    kf.Q = 0.0
    kf.predict()
    return np.array(kf.x), np.array(kf.P), captured["mapped"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def apply_theme(theme):
    c = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "turbo-ckf",
            "text.color": c["fg"],
            "axes.labelcolor": c["fg"],
            "axes.edgecolor": c["faint"],
            "axes.facecolor": "none",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "grid.color": c["faint"],
            "grid.linewidth": 0.6,
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "xtick.labelcolor": c["muted"],
            "ytick.labelcolor": c["muted"],
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
        }
    )
    return c


def ellipse(mean, L, k=1.0, n=200):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    return (np.asarray(mean)[:, None] + k * (L @ circle)).T


def save(fig, out_dir, name, theme):
    path = out_dir / f"{name}-{theme}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=200, metadata={"Date": None})
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def draw_tracking(out_dir, theme):
    c = apply_theme(theme)
    t, pos, vel, zs, xs, Ps = tracking_data()
    rmse_z = float(np.sqrt(np.mean((zs - pos) ** 2)))
    rmse_x = float(np.sqrt(np.mean((xs[:, 0] - pos) ** 2)))
    sd_pos = np.sqrt(Ps[:, 0, 0])
    sd_vel = np.sqrt(Ps[:, 1, 1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    ax1.axhline(0.0, color=c["faint"], linewidth=1.0)
    ax1.scatter(t, zs - pos, s=7, color=c["muted"], alpha=0.7, linewidths=0,
                label=f"measurement minus truth (RMSE {rmse_z:.2f} m)")
    ax1.fill_between(t, -2 * sd_pos, 2 * sd_pos, color=c["accent"], alpha=0.16, linewidth=0,
                     label="filter's own ±2σ (from P)")
    ax1.plot(t, xs[:, 0] - pos, color=c["accent"], linewidth=1.6,
             label=f"estimate minus truth (RMSE {rmse_x:.2f} m)")
    ax1.set_ylim(-1.8, 2.9)
    ax1.set_ylabel("position error (m)")
    ax1.set_title("Position: noisy readings in, a smoother estimate out")
    ax1.legend(loc="upper left", ncol=2, fontsize=8.5)

    ax2.fill_between(t, xs[:, 1] - 2 * sd_vel, xs[:, 1] + 2 * sd_vel, color=c["accent"],
                     alpha=0.16, linewidth=0)
    ax2.plot(t, vel, color=c["fg"], linewidth=1.2, label="true velocity")
    ax2.plot(t, xs[:, 1], color=c["accent"], linewidth=1.6, label="estimated velocity")
    ax2.set_ylim(-0.15, vel.max() + 0.6)
    ax2.set_ylabel("velocity (m/s)")
    ax2.set_xlabel("time (s)")
    ax2.set_title("Velocity: never measured, inferred from the motion model")
    ax2.legend(loc="upper right", ncol=2)
    fig.tight_layout(h_pad=1.6)
    stats = {"rmse_measurement": rmse_z, "rmse_estimate": rmse_x}
    return save(fig, out_dir, "tracking", theme), stats


def draw_cubature(out_dir, theme):
    c = apply_theme(theme)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.4))

    # (a) points from the Cholesky factor
    x = np.array([1.0, 2.0])
    P = np.array([[4.0, 1.2], [1.2, 1.0]])
    n = x.size
    pts = cubature_points_from_filter(x, P)
    L = np.linalg.cholesky(P)
    e1 = ellipse(x, L, 1.0)
    en = ellipse(x, L, np.sqrt(n))
    ax1.plot(e1[:, 0], e1[:, 1], color=c["muted"], linewidth=1.2, label="1σ ellipse of P")
    ax1.plot(en[:, 0], en[:, 1], color=c["muted"], linewidth=1.0, linestyle=(0, (4, 3)),
             label=f"√n σ ellipse (n = {n})")
    for k in range(n):
        tip = x + np.sqrt(n) * L[:, k]
        ax1.annotate("", xy=tip, xytext=x,
                     arrowprops={"arrowstyle": "-|>", "color": c["accent2"], "linewidth": 1.4})
        ax1.annotate(f"√n·L[:, {k}]", xy=(x + tip) / 2, xytext=(6, -12 if k == 0 else 4),
                     textcoords="offset points", color=c["accent2"], fontsize=9)
    ax1.scatter(pts[:, 0], pts[:, 1], s=42, color=c["accent"], zorder=3,
                label="cubature points, weight 1/(2n) each")
    for i, (px, py) in enumerate(pts):
        ax1.annotate(f"X{i + 1}", xy=(px, py), xytext=(6, 4), textcoords="offset points",
                     fontsize=9, color=c["fg"])
    ax1.scatter([x[0]], [x[1]], marker="x", s=50, color=c["fg"], zorder=3, label="mean x̂")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("2n points: x̂ ± √n times each column of chol(P)")
    ax1.legend(loc="lower right", fontsize=8.5)

    # (b) push them through a nonlinear model
    mean = np.array([1.0, np.pi / 2])
    Pp = np.diag([0.05**2, 0.6**2])
    cub_mean, cub_cov, mapped = nonlinear_moments(mean, Pp)
    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(mean, Pp, 20000)
    ys = fx_polar_to_cartesian(samples, 0.0)
    mc_mean = ys.mean(axis=0)
    mc_cov = np.cov(ys.T)
    plug = fx_polar_to_cartesian(mean[None, :], 0.0)[0]

    ax2.scatter(ys[:4000, 0], ys[:4000, 1], s=2, color=c["muted"], alpha=0.25, linewidths=0,
                rasterized=True, label="20,000 Monte Carlo samples")
    em = ellipse(mc_mean, np.linalg.cholesky(mc_cov))
    ec = ellipse(cub_mean, np.linalg.cholesky(cub_cov))
    ax2.plot(em[:, 0], em[:, 1], color=c["fg"], linewidth=1.2, linestyle=(0, (4, 3)),
             label=f"Monte Carlo mean and 1σ (y = {mc_mean[1]:.3f})")
    ax2.plot(ec[:, 0], ec[:, 1], color=c["accent"], linewidth=1.6,
             label=f"cubature mean and 1σ (y = {cub_mean[1]:.3f})")
    ax2.scatter(mapped[:, 0], mapped[:, 1], s=42, color=c["accent"], zorder=3)
    ax2.scatter([mc_mean[0]], [mc_mean[1]], marker="+", s=80, color=c["fg"], zorder=4)
    ax2.scatter([cub_mean[0]], [cub_mean[1]], marker="x", s=50, color=c["accent"], zorder=4)
    ax2.scatter([plug[0]], [plug[1]], s=60, facecolors="none", edgecolors=c["muted"],
                linewidths=1.4, zorder=4, label=f"plug in the mean (y = {plug[1]:.3f})")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("Through f: polar (r, θ) to Cartesian (x, y)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(loc="lower center", fontsize=8.5)
    fig.tight_layout(w_pad=2.0)
    stats = {
        "points": pts.tolist(),
        "cubature_mean": cub_mean.tolist(),
        "monte_carlo_mean": mc_mean.tolist(),
        "plug_in": plug.tolist(),
    }
    return save(fig, out_dir, "cubature", theme), stats


def draw_benchmark(out_dir, theme, report, machine):
    c = apply_theme(theme)
    rows = [b for b in report["benchmarks"] if b["name"] in BENCH_LABELS]
    order = list(BENCH_LABELS)
    rows.sort(key=lambda b: order.index(b["name"]))
    us = np.array([b["ns_per_step"] / 1000.0 for b in rows])
    base = us[0]
    labels = [BENCH_LABELS[b["name"]] for b in rows]
    colors = [c["muted"]] + [c["accent"]] * (len(rows) - 1)

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.set_axisbelow(True)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, us, color=colors, height=0.58)
    for yi, value in zip(y, us):
        note = f"{value:.1f} µs"
        if value != base:
            note += f"  ({base / value:.1f}x faster)"
        ax.text(value + base * 0.012, yi, note, va="center", fontsize=9, color=c["fg"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, labelcolor=c["fg"])
    ax.set_xlim(0, base * 1.22)
    ax.set_xlabel("time per predict + update step (µs, lower is better)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    env = report["environment"]
    cfg = report["config"]
    where = f"{machine}, " if machine else ""
    fig.text(
        0.0, -0.04,
        f"2-state constant-velocity filter, {cfg['benchmark_steps']:,} steps, median of "
        f"{cfg['benchmark_repeats']} runs. {where}Python {env['python']}, NumPy {env['numpy']}. "
        f"Report: docs/img/benchmark.json ({report['generated_at_utc'][:10]}).",
        fontsize=8, color=c["muted"], ha="left",
    )
    fig.tight_layout()
    return save(fig, out_dir, "benchmark", theme), dict(zip([b["name"] for b in rows], us.tolist()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    parser.add_argument("--bench", type=Path, default=ROOT / "docs" / "img" / "benchmark.json",
                        help="report written by verify_before_after.py --output")
    parser.add_argument("--machine", default=None, help='label for the chart, e.g. "Apple M4 Max"')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    report = json.loads(args.bench.read_text(encoding="utf-8")) if args.bench.exists() else None
    if report is None:
        print(f"no benchmark report at {args.bench}; skipping the benchmark chart")

    for theme in THEMES:
        path, stats = draw_tracking(args.out, theme)
        print(f"wrote {path.relative_to(ROOT)}  {stats}")
        path, stats = draw_cubature(args.out, theme)
        print(f"wrote {path.relative_to(ROOT)}  {stats}")
        if report is not None:
            path, stats = draw_benchmark(args.out, theme, report, args.machine)
            print(f"wrote {path.relative_to(ROOT)}  {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

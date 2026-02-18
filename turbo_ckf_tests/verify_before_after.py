"""Generate reproducible benchmark/parity reports and compare before/after runs."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_ckf import TurboCKF

try:
    from filterpy.kalman import CubatureKalmanFilter as FilterPyCKF
except Exception:  # pragma: no cover - optional dependency
    FilterPyCKF = None


def fx_vectorized(x, dt):
    if x.ndim == 2:
        out = x.copy()
        out[:, 0] = x[:, 0] + dt * x[:, 1]
        return out
    return np.array([x[0] + dt * x[1], x[1]], dtype=float)


def fx_pointwise(x, dt):
    return np.array([x[0] + dt * x[1], x[1]], dtype=float)


def hx_vectorized(x):
    if x.ndim == 2:
        return x[:, :1]
    return x[:1]


def hx_pointwise(x):
    return x[:1]


def make_filterpy():
    if FilterPyCKF is None:
        raise RuntimeError("filterpy is not available")
    kf = FilterPyCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_pointwise, fx=fx_pointwise)
    kf.x = np.array([0.0, 1.0], dtype=float)
    kf.P = np.eye(2) * 0.5
    kf.Q = np.eye(2) * 1e-3
    kf.R = np.eye(1) * 1e-2
    return kf


def make_turbo():
    ckf = TurboCKF(dim_x=2, dim_z=1, dt=0.1, hx=hx_vectorized, fx=fx_vectorized)
    ckf.x = np.array([0.0, 1.0], dtype=float)
    ckf.P = np.eye(2) * 0.5
    ckf.Q = np.eye(2) * 1e-3
    ckf.R = np.eye(1) * 1e-2
    return ckf


def run_predict_update(kf, steps: int):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        kf.predict()
        kf.update(z)


def run_turbo_model(ckf, steps: int):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        ckf.predict_standard_model("constant_velocity")
        ckf.update(z)


def run_turbo_model_ckf(ckf, steps: int):
    z = np.array([0.1], dtype=float)
    for _ in range(steps):
        ckf.predict_standard_model_ckf("constant_velocity")
        ckf.update(z)


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def run_parity_report(steps: int) -> dict:
    if FilterPyCKF is None:
        return {
            "enabled": False,
            "steps": int(steps),
            "reason": "filterpy not installed",
        }

    turbo = make_turbo()
    fp = make_filterpy()
    max_abs_state_diff = 0.0
    squared_error_sum = 0.0
    element_count = 0
    final_state_diff = np.zeros(2, dtype=float)

    for k in range(steps):
        z = np.array([0.05 * k], dtype=float)
        turbo.predict()
        turbo.update(z)
        fp.predict()
        fp.update(z)

        diff = _as_float_array(turbo.x) - _as_float_array(fp.x)
        final_state_diff = diff
        max_abs_state_diff = max(max_abs_state_diff, float(np.max(np.abs(diff))))
        squared_error_sum += float(np.dot(diff, diff))
        element_count += diff.size

    rmse = float(np.sqrt(squared_error_sum / max(1, element_count)))
    return {
        "enabled": True,
        "steps": int(steps),
        "max_abs_state_diff": max_abs_state_diff,
        "rmse_state_diff": rmse,
        "final_state_diff": final_state_diff.tolist(),
        "turbo_final_state": _as_float_array(turbo.x).tolist(),
        "filterpy_final_state": _as_float_array(fp.x).tolist(),
    }


def timed_case(
    case_name: str,
    factory: Callable[[], object],
    runner: Callable[[object, int], None],
    steps: int,
    repeats: int,
    warmup: int,
) -> dict:
    for _ in range(warmup):
        kf = factory()
        runner(kf, steps)

    samples = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            kf = factory()
            t0 = time.perf_counter()
            runner(kf, steps)
            samples.append(time.perf_counter() - t0)
    finally:
        if gc_enabled:
            gc.enable()

    arr = np.asarray(samples, dtype=float)
    median_s = float(np.median(arr))
    return {
        "name": case_name,
        "steps": int(steps),
        "repeats": int(repeats),
        "samples_s": [float(v) for v in arr],
        "mean_s": float(np.mean(arr)),
        "median_s": median_s,
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "std_s": float(np.std(arr, ddof=0)),
        "steps_per_second": float(steps / median_s) if median_s > 0.0 else None,
        "ns_per_step": float((median_s / steps) * 1e9) if steps > 0 else None,
    }


def run_benchmarks(steps: int, repeats: int, warmup: int) -> list[dict]:
    cases: list[tuple[str, Callable[[], object], Callable[[object, int], None]]] = [
        ("turbo_callback_predict_update", make_turbo, run_predict_update),
        ("turbo_standard_model_predict_update", make_turbo, run_turbo_model),
        ("turbo_standard_model_ckf_predict_update", make_turbo, run_turbo_model_ckf),
    ]
    if FilterPyCKF is not None:
        cases.insert(0, ("filterpy_predict_update", make_filterpy, run_predict_update))

    reports = []
    for case_name, factory, runner in cases:
        reports.append(timed_case(case_name, factory, runner, steps, repeats, warmup))
    return reports


def generate_report(steps: int, repeats: int, warmup: int, parity_steps: int) -> dict:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "filterpy_available": FilterPyCKF is not None,
        },
        "config": {
            "benchmark_steps": int(steps),
            "benchmark_repeats": int(repeats),
            "benchmark_warmup": int(warmup),
            "parity_steps": int(parity_steps),
        },
        "benchmarks": run_benchmarks(steps, repeats, warmup),
        "parity": run_parity_report(parity_steps),
    }


def compare_reports(baseline: dict, current: dict) -> dict:
    baseline_cases = {case["name"]: case for case in baseline.get("benchmarks", [])}
    current_cases = {case["name"]: case for case in current.get("benchmarks", [])}

    benchmark_comparison = {}
    for name, current_case in current_cases.items():
        baseline_case = baseline_cases.get(name)
        if baseline_case is None:
            continue

        baseline_median = float(baseline_case["median_s"])
        current_median = float(current_case["median_s"])
        speedup = baseline_median / current_median if current_median > 0.0 else None
        delta_pct = ((current_median - baseline_median) / baseline_median * 100.0) if baseline_median > 0.0 else None
        benchmark_comparison[name] = {
            "baseline_median_s": baseline_median,
            "current_median_s": current_median,
            "delta_s": current_median - baseline_median,
            "delta_pct": delta_pct,
            "speedup_vs_baseline": speedup,
        }

    baseline_parity = baseline.get("parity", {})
    current_parity = current.get("parity", {})
    parity_comparison: dict[str, object] = {"available": False}
    if baseline_parity.get("enabled") and current_parity.get("enabled"):
        base_max = float(baseline_parity["max_abs_state_diff"])
        cur_max = float(current_parity["max_abs_state_diff"])
        ratio = (cur_max / base_max) if base_max > 0.0 else None
        parity_comparison = {
            "available": True,
            "baseline_max_abs_state_diff": base_max,
            "current_max_abs_state_diff": cur_max,
            "delta_max_abs_state_diff": cur_max - base_max,
            "ratio_current_to_baseline": ratio,
        }

    return {
        "shared_benchmark_cases": sorted(benchmark_comparison.keys()),
        "missing_in_baseline": sorted(set(current_cases) - set(baseline_cases)),
        "missing_in_current": sorted(set(baseline_cases) - set(current_cases)),
        "benchmarks": benchmark_comparison,
        "parity": parity_comparison,
    }


def print_summary(report: dict):
    print("Verification Report")
    print(f"Generated (UTC): {report['generated_at_utc']}")
    cfg = report["config"]
    print(
        "Benchmark config:"
        f" steps={cfg['benchmark_steps']}, repeats={cfg['benchmark_repeats']}, warmup={cfg['benchmark_warmup']}"
    )
    print("")
    print("Benchmark medians:")
    for case in report["benchmarks"]:
        print(f"  {case['name']:<40} {case['median_s']:.6f}s  ({case['ns_per_step']:.1f} ns/step)")
    print("")

    parity = report["parity"]
    if parity.get("enabled"):
        print(
            f"Parity: enabled, steps={parity['steps']},"
            f" max_abs_state_diff={parity['max_abs_state_diff']:.3e}, rmse_state_diff={parity['rmse_state_diff']:.3e}"
        )
    else:
        print(f"Parity: disabled ({parity.get('reason', 'unknown reason')})")


def print_comparison(comparison: dict):
    print("")
    print("Before/After Comparison (relative to baseline):")
    for name in comparison["shared_benchmark_cases"]:
        item = comparison["benchmarks"][name]
        delta_pct = item["delta_pct"]
        pct_text = f"{delta_pct:+.2f}%" if delta_pct is not None else "n/a"
        speedup = item["speedup_vs_baseline"]
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
        print(
            f"  {name:<40} current={item['current_median_s']:.6f}s"
            f" baseline={item['baseline_median_s']:.6f}s"
            f" delta={pct_text} speedup={speedup_text}"
        )

    parity = comparison["parity"]
    if parity.get("available"):
        ratio = parity["ratio_current_to_baseline"]
        ratio_text = f"{ratio:.3f}x" if ratio is not None else "n/a"
        print(
            "  parity max_abs_state_diff:"
            f" current={parity['current_max_abs_state_diff']:.3e}"
            f" baseline={parity['baseline_max_abs_state_diff']:.3e}"
            f" ratio={ratio_text}"
        )
    else:
        print("  parity comparison: not available")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50_000, help="steps per benchmark run")
    parser.add_argument("--repeats", type=int, default=7, help="number of benchmark repeats")
    parser.add_argument("--warmup", type=int, default=1, help="number of benchmark warmup runs")
    parser.add_argument("--parity-steps", type=int, default=500, help="steps for parity run")
    parser.add_argument("--compare-to", type=Path, default=None, help="path to baseline JSON report")
    parser.add_argument("--output", type=Path, default=None, help="path to write JSON report")
    parser.add_argument("--json-stdout", action="store_true", help="print full JSON report to stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = generate_report(args.steps, args.repeats, args.warmup, args.parity_steps)

    if args.compare_to is not None:
        baseline = json.loads(args.compare_to.read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(baseline, report)

    print_summary(report)
    if "comparison" in report:
        print_comparison(report["comparison"])

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote report: {args.output}")

    if args.json_stdout:
        print("")
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

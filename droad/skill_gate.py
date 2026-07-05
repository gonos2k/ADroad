"""Forecast skill gate — compare a model's holdout forecast to a baseline while
holding it to the mass-audit and deviation budget.

What the gate actually enforces (hard failures):
  1. skill: Tsurf **RMSE** must not exceed the baseline by more than a tolerance,
  2. the accounting residual stays ~0 (P0, from droad.deviation), and
  3. the physics/deviation burden does not worsen vs. baseline (diagnostic step
     rate, over-melt count).

`forecast_metrics` also computes MAE, freeze-thaw accuracy, and cold RMSE, but
these are **report-only** — they are shown for interpretation, not gated. (RMSE
as the single hard skill gate is deliberate at this stage: simple and stable. A
run can therefore PASS with slightly worse MAE if its RMSE improved.)

This keeps "numerically fit" and "physically trustworthy" separate: a run that
lowers RMSE but blows up over-melt / overflow is flagged, not silently accepted.

Pure NumPy metric + gate logic (unit-testable without running any model). CLI
wiring lives in tools/report_skill_gate.py.
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping

import numpy as np


class SkillError(ValueError):
    """Raised on malformed skill inputs (shape/finite/schema)."""


def _as_series(name: str, x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise SkillError(f"{name} must be 1-D, got ndim {a.ndim}")
    if a.size == 0:
        raise SkillError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(a))):
        raise SkillError(f"{name} must be finite")
    return a


def forecast_metrics(pred, obs, *, freeze_thr: float = 0.0, cold_thr: float = 0.0) -> dict:
    """Skill metrics of `pred` vs `obs` over a forecast window (same length).

    - rmse / mae: surface-temperature error
    - freeze_thaw_accuracy: fraction of steps on the correct side of freeze_thr
      (a road-weather-relevant classification, not just magnitude)
    - cold_rmse / cold_n: rmse restricted to genuinely cold observations
      (obs < cold_thr); cold_rmse is None if there are none.
    """
    p = _as_series("pred", pred)
    o = _as_series("obs", obs)
    if p.shape != o.shape:
        raise SkillError(f"pred/obs length differ: {p.size} vs {o.size}")
    err = p - o
    cold = o < cold_thr
    cold_n = int(np.sum(cold))
    return {
        "n": int(p.size),
        "rmse": float(np.mean(err * err) ** 0.5),
        "mae": float(np.mean(np.abs(err))),
        "freeze_thaw_accuracy": float(np.mean((p >= freeze_thr) == (o >= freeze_thr))),
        "cold_n": cold_n,
        "cold_rmse": (float(np.mean(err[cold] ** 2) ** 0.5) if cold_n else None),
    }


def _finite_scalar(name: str, x) -> float:
    if isinstance(x, bool):
        raise SkillError(f"{name} must be numeric, not bool")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise SkillError(f"{name} must be a finite scalar, got {x!r}") from None
    if not np.isfinite(v):
        raise SkillError(f"{name} must be finite, got {x!r}")
    return v


def aggregate_metrics(metrics_list) -> dict:
    """Aggregate per-window forecast metrics into a stability summary. Used to
    judge a model across multiple periods, not one lucky window: rmse_mean and
    rmse_max (worst window) matter more than any single window."""
    if not isinstance(metrics_list, (list, tuple)) or not metrics_list:
        raise SkillError("metrics_list must be a non-empty sequence of metric mappings")
    rmses, ft = [], []
    for i, m in enumerate(metrics_list):
        if not isinstance(m, ABCMapping) or "rmse" not in m:
            raise SkillError(f"metrics_list[{i}] must be a metric mapping with 'rmse'")
        rmses.append(_finite_scalar(f"metrics_list[{i}].rmse", m["rmse"]))
        ft.append(_finite_scalar(f"metrics_list[{i}].freeze_thaw_accuracy",
                                 m.get("freeze_thaw_accuracy", 0.0)))
    n = len(rmses)
    return {"n_windows": n, "rmse_mean": sum(rmses) / n, "rmse_max": max(rmses),
            "rmse_min": min(rmses), "freeze_thaw_accuracy_mean": sum(ft) / n}


def degradation_ratio(holdout_rmse, train_rmse) -> float:
    """holdout RMSE / train RMSE — >1 means the model generalizes worse than it
    fit (overfitting signal), ~1 means it holds up. train_rmse must be positive."""
    h = _finite_scalar("holdout_rmse", holdout_rmse)
    t = _finite_scalar("train_rmse", train_rmse)
    if h < 0.0 or t < 0.0:
        raise SkillError("RMSE must be non-negative")
    if t <= 0.0:
        raise SkillError("train_rmse must be positive to form a degradation ratio")
    return h / t


def skill_gate(candidate: dict, baseline: dict, *, deviation=None, baseline_deviation=None,
               rmse_worse_frac: float = 0.0, residual_atol: float = 1e-9,
               rate_worse_abs: float = 0.0, over_melt_worse_abs: int = 0) -> tuple[bool, list[str]]:
    """Gate a candidate against a baseline. Returns (passed, reasons).

    skill: candidate RMSE must not exceed baseline RMSE by more than
    `rmse_worse_frac` (0.0 = must be <= baseline). accounting: if a deviation
    summary is given, its residual must be <= residual_atol. physics: if both
    deviation summaries are given, the candidate's diagnostic_steps_rate and
    over_melt_count must not exceed the baseline's by more than the given slack.
    """
    if not (isinstance(candidate, ABCMapping) and isinstance(baseline, ABCMapping)):
        raise SkillError("candidate and baseline must be metric mappings")
    for m, nm in ((candidate, "candidate"), (baseline, "baseline")):
        if "rmse" not in m:
            raise SkillError(f"{nm} missing 'rmse'")
    # validate scalars — a NaN rmse would make `nan > baseline` False and false-PASS
    c_rmse = _finite_scalar("candidate.rmse", candidate["rmse"])
    b_rmse = _finite_scalar("baseline.rmse", baseline["rmse"])
    rmse_worse_frac = _finite_scalar("rmse_worse_frac", rmse_worse_frac)
    residual_atol = _finite_scalar("residual_atol", residual_atol)
    rate_worse_abs = _finite_scalar("rate_worse_abs", rate_worse_abs)
    over_melt_worse_abs = _finite_scalar("over_melt_worse_abs", over_melt_worse_abs)
    if c_rmse < 0.0 or b_rmse < 0.0 or residual_atol < 0.0:
        raise SkillError("rmse and residual_atol must be non-negative")
    reasons = []
    if c_rmse > b_rmse * (1.0 + rmse_worse_frac):
        reasons.append(f"forecast RMSE {c_rmse:.4f} worse than baseline {b_rmse:.4f}")
    if deviation is not None:
        resid = _finite_scalar("deviation.max_primary_residual", deviation["max_primary_residual"])
        if resid > residual_atol:
            reasons.append(f"accounting residual {resid:.3e} > {residual_atol:.0e}")
        if baseline_deviation is not None:
            c_rate = _finite_scalar("deviation.diagnostic_steps_rate", deviation["diagnostic_steps_rate"])
            b_rate = _finite_scalar("baseline_deviation.diagnostic_steps_rate", baseline_deviation["diagnostic_steps_rate"])
            c_om = _finite_scalar("deviation.over_melt_count", deviation["over_melt_count"])
            b_om = _finite_scalar("baseline_deviation.over_melt_count", baseline_deviation["over_melt_count"])
            if c_rate > b_rate + rate_worse_abs:
                reasons.append("diagnostic_steps_rate worse than baseline")
            if c_om > b_om + over_melt_worse_abs:
                reasons.append("over_melt_count worse than baseline")
    return (not reasons), reasons


_DEV_KEYS = ("max_primary_residual", "diagnostic_steps_rate", "over_melt_count", "overflow_count")


def diagnostics_delta(candidate_dev, baseline_dev) -> dict:
    """Diagnostics-aware comparison: how a candidate's physics burden differs from
    a baseline's (e.g. did DA lower RMSE but raise over-melt?). Returns per-key
    deltas plus `physics_worse` = any burden increased. Missing keys count as 0."""
    for d, nm in ((candidate_dev, "candidate_dev"), (baseline_dev, "baseline_dev")):
        if not isinstance(d, ABCMapping):
            raise SkillError(f"{nm} must be a deviation summary mapping")
    delta = {}
    for k in _DEV_KEYS:                      # validate values (NaN/str -> SkillError, not bad delta)
        c = _finite_scalar(f"candidate_dev.{k}", candidate_dev.get(k, 0))
        b = _finite_scalar(f"baseline_dev.{k}", baseline_dev.get(k, 0))
        delta[f"delta_{k}"] = c - b
    delta["physics_worse"] = bool(
        delta["delta_diagnostic_steps_rate"] > 0
        or delta["delta_over_melt_count"] > 0
        or delta["delta_overflow_count"] > 0)
    return delta


def promotion_gate(*, n_cases, windows_beat_baseline, deviation=None,
                   baseline_deviation=None, min_cases: int = 3, residual_atol: float = 1e-9
                   ) -> tuple[str, list[str]]:
    """Promote a model from report-only to trusted only when the WHOLE evidence
    holds (design §11): enough distinct cases, skill beats baseline in every
    window, accounting residual clean, and physics burden not worse than baseline.

    Returns ("PROMOTE" | "REPORT_ONLY", reasons). On a single fixture this
    correctly returns REPORT_ONLY (insufficient cases) — the report-only policy
    is thus an executable gate, not a doc note.
    """
    n_cases = int(_finite_scalar("n_cases", n_cases))
    reasons = []
    if n_cases < min_cases:
        reasons.append(f"insufficient cases: {n_cases} < {min_cases} (report-only)")
    if not windows_beat_baseline:
        reasons.append("does not beat baseline in every window")
    if deviation is not None:
        if deviation["max_primary_residual"] > residual_atol:
            reasons.append(
                f"accounting residual {deviation['max_primary_residual']:.3e} > {residual_atol:.0e}")
        if baseline_deviation is not None and diagnostics_delta(deviation, baseline_deviation)["physics_worse"]:
            reasons.append("physics burden worse than baseline")
    return ("PROMOTE" if not reasons else "REPORT_ONLY"), reasons


_COLUMNS = ("model", "n", "rmse", "mae", "freeze_thaw_accuracy", "cold_n", "cold_rmse", "gate")


def _require_columns(row):
    if not isinstance(row, ABCMapping):
        raise SkillError("skill row must be a mapping")
    missing = set(_COLUMNS) - set(row)
    if missing:
        raise SkillError(f"skill row missing columns: {sorted(missing, key=str)}")


def skill_report_csv(rows) -> str:
    """Machine-readable CSV (raw precision)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_COLUMNS)
    for r in rows:
        _require_columns(r)
        w.writerow([r[c] for c in _COLUMNS])
    return buf.getvalue()


def _md_cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def skill_report_markdown(rows, title: str = "Forecast Skill Gate") -> str:
    head = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [f"# {_md_cell(title)}", "",
             "skill = 예측 정확도(낮을수록 좋음). gate = skill + 회계 residual + diagnostics 부담 종합.",
             "", head, sep]
    for r in rows:
        _require_columns(r)
        lines.append("| " + " | ".join(_md_cell(r[c]) for c in _COLUMNS) + " |")
    return "\n".join(lines) + "\n"

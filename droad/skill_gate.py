"""Forecast skill gate — compare a model's holdout forecast to a baseline while
holding it to the mass-audit and deviation budget.

A "good" forecast, per design, is one where:
  1. skill improves (or is not worse) — Tsurf RMSE/MAE/freeze-thaw accuracy,
  2. the accounting residual stays ~0 (P0, from droad.deviation), and
  3. the physics/deviation burden does not worsen vs. baseline (diagnostic step
     rate, over-melt count).

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
    reasons = []
    if candidate["rmse"] > baseline["rmse"] * (1.0 + rmse_worse_frac):
        reasons.append(f"forecast RMSE {candidate['rmse']:.4f} worse than baseline {baseline['rmse']:.4f}")
    if deviation is not None:
        if deviation["max_primary_residual"] > residual_atol:
            reasons.append(
                f"accounting residual {deviation['max_primary_residual']:.3e} > {residual_atol:.0e}")
        if baseline_deviation is not None:
            if deviation["diagnostic_steps_rate"] > baseline_deviation["diagnostic_steps_rate"] + rate_worse_abs:
                reasons.append("diagnostic_steps_rate worse than baseline")
            if deviation["over_melt_count"] > baseline_deviation["over_melt_count"] + over_melt_worse_abs:
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
    delta = {f"delta_{k}": (candidate_dev.get(k, 0) - baseline_dev.get(k, 0)) for k in _DEV_KEYS}
    delta["physics_worse"] = bool(
        delta["delta_diagnostic_steps_rate"] > 0
        or delta["delta_over_melt_count"] > 0
        or delta["delta_overflow_count"] > 0)
    return delta


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

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
    # reject bool/str/bytes/object arrays BEFORE float coercion, matching the scalar
    # policy (True->1.0, "1.0"->1.0 would otherwise pass silently as a metric series).
    arr = np.asarray(x)
    if arr.dtype.kind not in ("i", "u", "f"):
        raise SkillError(f"{name} must be a numeric array, not bool/string/object")
    a = arr.astype(float)
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
    # thresholds must be finite: a NaN freeze_thr makes every (x>=nan) False, faking
    # freeze_thaw_accuracy ~1.0; a NaN cold_thr silently empties the cold subset.
    freeze_thr = _finite_scalar("freeze_thr", freeze_thr)
    cold_thr = _finite_scalar("cold_thr", cold_thr)
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
    if isinstance(x, (bool, np.bool_)):       # numpy bool is also not a metric value
        raise SkillError(f"{name} must be numeric, not bool")
    if isinstance(x, (str, bytes)):           # "1.0" would float() silently — match ledger policy
        raise SkillError(f"{name} must be numeric, not string")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise SkillError(f"{name} must be a finite scalar, got {x!r}") from None
    if not np.isfinite(v):
        raise SkillError(f"{name} must be finite, got {x!r}")
    return v


def _int_count(name: str, x) -> int:
    """A count that must be a non-negative WHOLE number (2.9 cases is malformed input)."""
    v = _finite_scalar(name, x)
    if v < 0 or int(v) != v:
        raise SkillError(f"{name} must be a non-negative integer, got {x!r}")
    return int(v)


# the deviation-budget fields the gate/promotion logic reads — one schema, one bar
_DEV_KEYS = ("max_primary_residual", "diagnostic_steps_rate", "over_melt_count", "overflow_count")


def _require_dev_summary(d, name: str) -> dict:
    """Validate a deviation summary to the same bar as the ledger/deviation layer:
    a mapping carrying every burden key as a finite, NON-NEGATIVE scalar. Rejects
    bad public input with SkillError instead of a raw KeyError / a silent false-PASS
    (a negative burden or NaN residual would otherwise slip through the comparisons)."""
    if not isinstance(d, ABCMapping):
        raise SkillError(f"{name} must be a deviation summary mapping")
    vals = {}
    for k in _DEV_KEYS:
        if k not in d:
            raise SkillError(f"{name} missing {k}")
        v = _finite_scalar(f"{name}.{k}", d[k])
        if v < 0.0:
            raise SkillError(f"{name}.{k} must be non-negative")
        # enforce semantic ranges: counts are whole numbers, a step-rate is a fraction
        if k in ("over_melt_count", "overflow_count") and int(v) != v:
            raise SkillError(f"{name}.{k} must be a whole count, got {d[k]!r}")
        if k == "diagnostic_steps_rate" and v > 1.0:
            raise SkillError(f"{name}.{k} must be in [0, 1], got {d[k]!r}")
        vals[k] = v
    return vals


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
        r = _finite_scalar(f"metrics_list[{i}].rmse", m["rmse"])
        if r < 0.0:
            raise SkillError(f"metrics_list[{i}].rmse must be non-negative")
        rmses.append(r)
        if "freeze_thaw_accuracy" not in m:      # don't silently average in a fake 0.0
            raise SkillError(f"metrics_list[{i}] missing freeze_thaw_accuracy")
        acc = _finite_scalar(f"metrics_list[{i}].freeze_thaw_accuracy",
                             m["freeze_thaw_accuracy"])
        if not (0.0 <= acc <= 1.0):          # it's a fraction; a public caller could pass 2.0
            raise SkillError(f"metrics_list[{i}].freeze_thaw_accuracy must be in [0, 1]")
        ft.append(acc)
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
               rate_worse_abs: float = 0.0, over_melt_worse_abs: int = 0,
               overflow_worse_abs: int = 0) -> tuple[bool, list[str]]:
    """Gate a candidate against a baseline. Returns (passed, reasons).

    skill: candidate RMSE must not exceed baseline RMSE by more than
    `rmse_worse_frac` (0.0 = must be <= baseline). accounting: if a deviation
    summary is given, its residual must be <= residual_atol. physics: if both
    deviation summaries are given, the candidate's diagnostic_steps_rate,
    over_melt_count AND overflow_count must not exceed the baseline's by more than
    the given slack — the same three burdens diagnostics_delta().physics_worse
    watches, so the gate and that flag can never disagree.
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
    # over_melt/overflow are COUNTS: their slack must be a whole non-negative number
    over_melt_worse_abs = _int_count("over_melt_worse_abs", over_melt_worse_abs)
    overflow_worse_abs = _int_count("overflow_worse_abs", overflow_worse_abs)
    if c_rmse < 0.0 or b_rmse < 0.0 or residual_atol < 0.0:
        raise SkillError("rmse and residual_atol must be non-negative")
    # slacks widen the gate; a negative slack silently makes it STRICTER than
    # intended (API misuse), so reject rather than honor it.
    if rmse_worse_frac < 0.0 or rate_worse_abs < 0.0:
        raise SkillError("gate tolerances/slacks must be non-negative")
    # baseline_deviation is only consulted inside the `deviation` block, so passing it
    # alone silently does nothing — flag the likely caller mistake (matches promotion_gate).
    if baseline_deviation is not None and deviation is None:
        raise SkillError("baseline_deviation requires deviation")
    reasons = []
    if c_rmse > b_rmse * (1.0 + rmse_worse_frac):
        reasons.append(f"forecast RMSE {c_rmse:.4f} worse than baseline {b_rmse:.4f}")
    if deviation is not None:
        dv = _require_dev_summary(deviation, "deviation")   # schema + finite + non-negative
        if dv["max_primary_residual"] > residual_atol:
            reasons.append(f"accounting residual {dv['max_primary_residual']:.3e} > {residual_atol:.0e}")
        if baseline_deviation is not None:
            bv = _require_dev_summary(baseline_deviation, "baseline_deviation")
            if dv["diagnostic_steps_rate"] > bv["diagnostic_steps_rate"] + rate_worse_abs:
                reasons.append("diagnostic_steps_rate worse than baseline")
            if dv["over_melt_count"] > bv["over_melt_count"] + over_melt_worse_abs:
                reasons.append("over_melt_count worse than baseline")
            if dv["overflow_count"] > bv["overflow_count"] + overflow_worse_abs:
                reasons.append("overflow_count worse than baseline")
    return (not reasons), reasons


def diagnostics_delta(candidate_dev, baseline_dev) -> dict:
    """Diagnostics-aware comparison: how a candidate's physics burden differs from
    a baseline's (e.g. did DA lower RMSE but raise over-melt?). Returns per-key
    deltas plus `physics_worse` = any burden increased.

    Unlike skill_gate()/promotion_gate() — which require a FULL deviation summary via
    _require_dev_summary — this helper is deliberately permissive for ad-hoc
    comparison: a missing burden key counts as 0."""
    for d, nm in ((candidate_dev, "candidate_dev"), (baseline_dev, "baseline_dev")):
        if not isinstance(d, ABCMapping):
            raise SkillError(f"{nm} must be a deviation summary mapping")
    delta = {}
    for k in _DEV_KEYS:                      # validate values (NaN/str -> SkillError, not bad delta)
        c = _finite_scalar(f"candidate_dev.{k}", candidate_dev.get(k, 0))
        b = _finite_scalar(f"baseline_dev.{k}", baseline_dev.get(k, 0))
        # residual/rate/counts are magnitudes from a deviation budget: a negative
        # value means a corrupted summary, not a real burden — reject it.
        if c < 0.0 or b < 0.0:
            raise SkillError(f"{k} must be non-negative in a deviation summary")
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
    # promotion is a high-stakes verdict — validate every input to the same bar as
    # skill_gate (a string "False" is truthy; a NaN/negative residual would false-PROMOTE;
    # a fractional case count is malformed).
    n_cases = _int_count("n_cases", n_cases)
    min_cases = _int_count("min_cases", min_cases)
    residual_atol = _finite_scalar("residual_atol", residual_atol)
    if min_cases <= 0 or residual_atol < 0.0:
        raise SkillError("min_cases must be positive and residual_atol non-negative")
    if not isinstance(windows_beat_baseline, (bool, np.bool_)):
        raise SkillError("windows_beat_baseline must be bool")
    if baseline_deviation is not None and deviation is None:
        raise SkillError("baseline_deviation requires deviation")   # likely caller mistake
    reasons = []
    if n_cases < min_cases:
        reasons.append(f"insufficient cases: {n_cases} < {min_cases} (report-only)")
    if not windows_beat_baseline:
        reasons.append("does not beat baseline in every window")
    if deviation is not None:
        dv = _require_dev_summary(deviation, "deviation")   # schema + finite + non-negative residual
        if dv["max_primary_residual"] > residual_atol:
            reasons.append(f"accounting residual {dv['max_primary_residual']:.3e} > {residual_atol:.0e}")
        if baseline_deviation is not None:
            # strict schema here too — promotion is high-stakes, so don't lean on the
            # permissive missing=0 policy of diagnostics_delta for the baseline.
            _require_dev_summary(baseline_deviation, "baseline_deviation")
            if diagnostics_delta(deviation, baseline_deviation)["physics_worse"]:
                reasons.append("physics burden worse than baseline")
    return ("PROMOTE" if not reasons else "REPORT_ONLY"), reasons


_COLUMNS = ("model", "n", "rmse", "mae", "freeze_thaw_accuracy", "cold_n", "cold_rmse", "gate")


def _require_columns(row):
    if not isinstance(row, ABCMapping):
        raise SkillError("skill row must be a mapping")
    missing = set(_COLUMNS) - set(row)
    if missing:
        raise SkillError(f"skill row missing columns: {sorted(missing, key=str)}")
    # numeric columns must not just be finite but semantically valid (same bar as the
    # deviation report) — counts are whole non-negative, errors non-negative, and
    # freeze_thaw_accuracy a fraction — so no corrupted row is silently serialized.
    _int_count("row[n]", row["n"])
    cold_n = _int_count("row[cold_n]", row["cold_n"])
    for c in ("rmse", "mae"):
        if _finite_scalar(f"row[{c}]", row[c]) < 0.0:
            raise SkillError(f"row[{c}] must be non-negative")
    acc = _finite_scalar("row[freeze_thaw_accuracy]", row["freeze_thaw_accuracy"])
    if not (0.0 <= acc <= 1.0):
        raise SkillError("row[freeze_thaw_accuracy] must be in [0, 1]")
    # cold_rmse ↔ cold_n consistency (matches forecast_metrics: None iff no cold obs)
    if cold_n == 0 and row["cold_rmse"] is not None:
        raise SkillError("row[cold_rmse] must be None when cold_n == 0")
    if cold_n > 0:
        if row["cold_rmse"] is None:
            raise SkillError("row[cold_rmse] required when cold_n > 0")
        if _finite_scalar("row[cold_rmse]", row["cold_rmse"]) < 0.0:
            raise SkillError("row[cold_rmse] must be non-negative")
    # model/gate are report labels: must be non-empty strings (so a blank/None
    # doesn't produce an anonymous, un-attributable row).
    for c in ("model", "gate"):
        if not str(row[c]).strip():
            raise SkillError(f"row[{c}] must be a non-empty string")


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

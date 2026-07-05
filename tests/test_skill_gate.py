"""Forecast skill metrics + gate: correctness, defensive input, serialization."""
import pytest

from droad.skill_gate import (
    SkillError, forecast_metrics, skill_gate, skill_report_csv, skill_report_markdown,
    diagnostics_delta, aggregate_metrics, degradation_ratio, promotion_gate,
)


def test_forecast_metrics_values():
    pred = [0.0, 1.0, 2.0, -1.0]
    obs = [0.0, 0.0, 0.0, 0.0]
    m = forecast_metrics(pred, obs)
    assert m["n"] == 4
    assert m["rmse"] == pytest.approx((sum(x * x for x in pred) / 4) ** 0.5)
    assert m["mae"] == pytest.approx(1.0)
    # freeze_thr=0.0: pred>=0 for [0,1,2] matches obs>=0; -1.0 mismatches obs 0.0
    assert m["freeze_thaw_accuracy"] == pytest.approx(0.75)
    assert m["cold_n"] == 0 and m["cold_rmse"] is None


def test_forecast_metrics_cold_subset():
    m = forecast_metrics([-2.0, 5.0], [-3.0, 5.0], cold_thr=0.0)
    assert m["cold_n"] == 1                          # only obs -3.0 < 0
    assert m["cold_rmse"] == pytest.approx(1.0)      # |-2 - -3| = 1


def test_forecast_metrics_defensive():
    with pytest.raises(SkillError):
        forecast_metrics([1.0, 2.0], [1.0])          # length mismatch
    with pytest.raises(SkillError):
        forecast_metrics([], [])                     # empty
    with pytest.raises(SkillError):
        forecast_metrics([1.0, float("nan")], [1.0, 2.0])   # non-finite
    with pytest.raises(SkillError):
        forecast_metrics([[1.0, 2.0]], [[1.0, 2.0]])        # 2-D


def test_skill_gate_skill_and_audit():
    cand = {"rmse": 0.2}
    base = {"rmse": 5.0}
    ok, reasons = skill_gate(cand, base)
    assert ok and reasons == []                      # candidate beats baseline

    ok2, reasons2 = skill_gate({"rmse": 6.0}, base)  # worse skill
    assert not ok2 and any("RMSE" in r for r in reasons2)

    dev = {"max_primary_residual": 1e-3, "diagnostic_steps_rate": 0.0,
           "over_melt_count": 0, "overflow_count": 0}
    ok3, reasons3 = skill_gate(cand, base, deviation=dev)   # skill ok but residual leaks
    assert not ok3 and any("residual" in r for r in reasons3)


def test_skill_gate_diagnostics_burden():
    cand, base = {"rmse": 0.2}, {"rmse": 5.0}
    dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.10,
           "over_melt_count": 9, "overflow_count": 0}
    base_dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.01,
                "over_melt_count": 0, "overflow_count": 0}
    ok, reasons = skill_gate(cand, base, deviation=dev, baseline_deviation=base_dev)
    assert not ok
    assert any("diagnostic_steps_rate" in r for r in reasons)
    assert any("over_melt" in r for r in reasons)


def test_skill_gate_rejects_bad_metrics():
    with pytest.raises(SkillError):
        skill_gate({"rmse": 0.2}, {"mae": 1.0})      # baseline missing rmse
    with pytest.raises(SkillError):
        skill_gate(None, {"rmse": 1.0})


def test_skill_gate_rejects_nan_rmse():
    with pytest.raises(SkillError):                  # NaN rmse would false-PASS the > comparison
        skill_gate({"rmse": float("nan")}, {"rmse": 0.2})
    with pytest.raises(SkillError):
        skill_gate({"rmse": 0.2}, {"rmse": float("inf")})
    dev = {"max_primary_residual": float("nan"), "diagnostic_steps_rate": 0.0,
           "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):                  # NaN residual would false-PASS
        skill_gate({"rmse": 0.2}, {"rmse": 5.0}, deviation=dev)


def test_diagnostics_delta_rejects_nonfinite_values():
    base = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
            "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):
        diagnostics_delta({**base, "over_melt_count": float("nan")}, base)


def test_diagnostics_delta():
    base = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.01,
            "over_melt_count": 0, "overflow_count": 0}
    # DA lowered nothing physical but raised over-melt -> physics_worse
    worse = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.01,
             "over_melt_count": 5, "overflow_count": 0}
    d = diagnostics_delta(worse, base)
    assert d["delta_over_melt_count"] == 5
    assert d["physics_worse"] is True
    # identical burden -> not worse
    d2 = diagnostics_delta(base, base)
    assert d2["physics_worse"] is False
    with pytest.raises(SkillError):
        diagnostics_delta(None, base)


def test_aggregate_metrics():
    ms = [{"rmse": 0.2, "freeze_thaw_accuracy": 0.99},
          {"rmse": 0.4, "freeze_thaw_accuracy": 0.97},
          {"rmse": 0.3, "freeze_thaw_accuracy": 0.98}]
    a = aggregate_metrics(ms)
    assert a["n_windows"] == 3
    assert a["rmse_mean"] == pytest.approx(0.3)
    assert a["rmse_max"] == pytest.approx(0.4)          # worst window
    assert a["rmse_min"] == pytest.approx(0.2)
    with pytest.raises(SkillError):
        aggregate_metrics([])                           # empty
    with pytest.raises(SkillError):
        aggregate_metrics([{"mae": 0.1}])               # missing rmse


def test_degradation_ratio():
    assert degradation_ratio(0.4, 0.2) == pytest.approx(2.0)   # holdout 2x worse -> overfit signal
    assert degradation_ratio(0.2, 0.2) == pytest.approx(1.0)
    with pytest.raises(SkillError):
        degradation_ratio(0.4, 0.0)                     # train_rmse must be positive
    with pytest.raises(SkillError):
        degradation_ratio(float("nan"), 0.2)


def test_promotion_gate_report_only_on_single_case():
    dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.004,
           "over_melt_count": 0, "overflow_count": 0}
    # single fixture, all windows beat baseline, clean accounting -> still report-only
    v, reasons = promotion_gate(n_cases=1, windows_beat_baseline=True, deviation=dev)
    assert v == "REPORT_ONLY"
    assert any("insufficient cases" in r for r in reasons)

    # enough cases + all conditions hold -> promote
    v2, r2 = promotion_gate(n_cases=3, windows_beat_baseline=True, deviation=dev)
    assert v2 == "PROMOTE" and r2 == []

    # enough cases but a window loses -> report-only
    v3, r3 = promotion_gate(n_cases=3, windows_beat_baseline=False, deviation=dev)
    assert v3 == "REPORT_ONLY" and any("every window" in r for r in r3)


def test_promotion_gate_blocks_on_residual_and_physics():
    base = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.004,
            "over_melt_count": 0, "overflow_count": 0}
    leak = {**base, "max_primary_residual": 1e-3}
    v, reasons = promotion_gate(n_cases=5, windows_beat_baseline=True, deviation=leak)
    assert v == "REPORT_ONLY" and any("residual" in r for r in reasons)

    worse = {**base, "over_melt_count": 9}
    v2, r2 = promotion_gate(n_cases=5, windows_beat_baseline=True,
                            deviation=worse, baseline_deviation=base)
    assert v2 == "REPORT_ONLY" and any("physics" in r for r in r2)


def test_skill_gate_gates_overflow_like_diagnostics_delta():
    # candidate keeps over-melt & rate flat but raises overflow — must FAIL, matching
    # diagnostics_delta().physics_worse (gate and flag can't disagree).
    cand, base = {"rmse": 0.2}, {"rmse": 5.0}
    dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.01,
           "over_melt_count": 0, "overflow_count": 7}
    base_dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.01,
                "over_melt_count": 0, "overflow_count": 0}
    ok, reasons = skill_gate(cand, base, deviation=dev, baseline_deviation=base_dev)
    assert not ok and any("overflow_count" in r for r in reasons)
    assert diagnostics_delta(dev, base_dev)["physics_worse"] is True   # both agree


def test_skill_gate_rejects_bad_deviation_summary():
    cand, base = {"rmse": 0.2}, {"rmse": 5.0}
    with pytest.raises(SkillError):                  # missing overflow_count key
        skill_gate(cand, base, deviation={"max_primary_residual": 0.0,
                                          "diagnostic_steps_rate": 0.0, "over_melt_count": 0})
    with pytest.raises(SkillError):                  # negative burden
        skill_gate(cand, base, deviation={"max_primary_residual": 0.0,
                   "diagnostic_steps_rate": 0.0, "over_melt_count": -1, "overflow_count": 0})
    with pytest.raises(SkillError):                  # not a mapping
        skill_gate(cand, base, deviation=[1, 2, 3])


def test_promotion_gate_rejects_fractional_cases():
    dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
           "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):                  # 2.9 cases is malformed
        promotion_gate(n_cases=2.9, windows_beat_baseline=True, deviation=dev)
    with pytest.raises(SkillError):                  # negative residual = corrupted summary
        promotion_gate(n_cases=5, windows_beat_baseline=True,
                       deviation={**dev, "max_primary_residual": -1.0})


def test_skill_gate_rejects_negative_slack():
    cand, base = {"rmse": 0.2}, {"rmse": 5.0}
    for bad in ("rmse_worse_frac", "rate_worse_abs", "over_melt_worse_abs", "overflow_worse_abs"):
        with pytest.raises(SkillError):          # negative slack silently tightens the gate
            skill_gate(cand, base, **{bad: -0.1})


def test_diagnostics_delta_rejects_negative_burden():
    base = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
            "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):              # a negative count = corrupted summary
        diagnostics_delta({**base, "over_melt_count": -1}, base)


def test_aggregate_metrics_rejects_out_of_range_accuracy():
    with pytest.raises(SkillError):              # freeze_thaw_accuracy is a fraction in [0,1]
        aggregate_metrics([{"rmse": 0.2, "freeze_thaw_accuracy": 2.0}])
    with pytest.raises(SkillError):
        aggregate_metrics([{"rmse": -0.1, "freeze_thaw_accuracy": 0.5}])


def test_promotion_gate_validates_inputs():
    dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
           "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):              # "False" string is truthy -> must be bool
        promotion_gate(n_cases=3, windows_beat_baseline="False", deviation=dev)
    with pytest.raises(SkillError):
        promotion_gate(n_cases=3, windows_beat_baseline=True, min_cases=0, deviation=dev)
    with pytest.raises(SkillError):              # NaN residual would false-PROMOTE
        promotion_gate(n_cases=5, windows_beat_baseline=True,
                       deviation={**dev, "max_primary_residual": float("nan")})


def test_skill_gate_baseline_dev_requires_deviation():
    base_dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
                "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):              # baseline_deviation alone is silently ignored -> flag it
        skill_gate({"rmse": 0.2}, {"rmse": 5.0}, baseline_deviation=base_dev)


def test_promotion_gate_baseline_dev_requires_deviation():
    base_dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
                "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):              # baseline_deviation without deviation = mistake
        promotion_gate(n_cases=5, windows_beat_baseline=True, baseline_deviation=base_dev)


def test_promotion_gate_validates_baseline_deviation_schema():
    dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
           "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):              # baseline missing overflow_count -> strict reject
        promotion_gate(n_cases=5, windows_beat_baseline=True, deviation=dev,
                       baseline_deviation={"max_primary_residual": 0.0,
                                           "diagnostic_steps_rate": 0.0, "over_melt_count": 0})


def test_skill_gate_rejects_fractional_count_slack():
    cand, base = {"rmse": 0.2}, {"rmse": 5.0}
    with pytest.raises(SkillError):              # over_melt_worse_abs is a count -> whole number
        skill_gate(cand, base, over_melt_worse_abs=0.5)
    with pytest.raises(SkillError):
        skill_gate(cand, base, overflow_worse_abs=1.5)


def test_finite_scalar_rejects_numpy_bool():
    import numpy as np
    with pytest.raises(SkillError):
        skill_gate({"rmse": np.bool_(True)}, {"rmse": 0.2})   # np.bool_ is not a metric


def test_finite_scalar_rejects_string_numeric():
    with pytest.raises(SkillError):              # "1.0" would float() silently — reject like ledger
        skill_gate({"rmse": "1.0"}, {"rmse": 0.2})


def test_require_dev_summary_enforces_count_and_rate_ranges():
    ok = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.0,
          "over_melt_count": 0, "overflow_count": 0}
    with pytest.raises(SkillError):              # count must be a whole number
        skill_gate({"rmse": 0.2}, {"rmse": 5.0}, deviation={**ok, "over_melt_count": 0.5})
    with pytest.raises(SkillError):              # step-rate is a fraction in [0,1]
        skill_gate({"rmse": 0.2}, {"rmse": 5.0}, deviation={**ok, "diagnostic_steps_rate": 2.0})


def test_skill_report_serialization():
    row = {"model": "default", "n": 100, "rmse": 1 / 3, "mae": 0.1,
           "freeze_thaw_accuracy": 0.99, "cold_n": 5, "cold_rmse": 0.2, "gate": "PASS"}
    csv = skill_report_csv([row])
    assert csv.splitlines()[0].startswith("model,")
    assert repr(1 / 3) in csv or str(1 / 3) in csv    # raw precision
    md = skill_report_markdown([row])
    assert "default" in md and "gate" in md
    with pytest.raises(SkillError):
        skill_report_csv([{"model": "x"}])           # missing columns

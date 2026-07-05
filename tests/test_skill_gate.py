"""Forecast skill metrics + gate: correctness, defensive input, serialization."""
import pytest

from droad.skill_gate import (
    SkillError, forecast_metrics, skill_gate, skill_report_csv, skill_report_markdown,
    diagnostics_delta,
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

    dev = {"max_primary_residual": 1e-3, "diagnostic_steps_rate": 0.0, "over_melt_count": 0}
    ok3, reasons3 = skill_gate(cand, base, deviation=dev)   # skill ok but residual leaks
    assert not ok3 and any("residual" in r for r in reasons3)


def test_skill_gate_diagnostics_burden():
    cand, base = {"rmse": 0.2}, {"rmse": 5.0}
    dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.10, "over_melt_count": 9}
    base_dev = {"max_primary_residual": 0.0, "diagnostic_steps_rate": 0.01, "over_melt_count": 0}
    ok, reasons = skill_gate(cand, base, deviation=dev, baseline_deviation=base_dev)
    assert not ok
    assert any("diagnostic_steps_rate" in r for r in reasons)
    assert any("over_melt" in r for r in reasons)


def test_skill_gate_rejects_bad_metrics():
    with pytest.raises(SkillError):
        skill_gate({"rmse": 0.2}, {"mae": 1.0})      # baseline missing rmse
    with pytest.raises(SkillError):
        skill_gate(None, {"rmse": 1.0})


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

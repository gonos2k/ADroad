"""A0 full-model multi-window: pure aggregation logic (fast) + jax-marked smoke."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da_fullmodel_multi import (  # noqa: E402
    _case_row, summarize_multi, _COLS,
)


def _dev(rate=0.0, resid=0.0):
    return {"diagnostic_steps_rate": rate, "max_primary_residual": resid,
            "over_melt_count": 0, "overflow_count": 0}


def _result(k0, da_rmse, bg_rmse, gate_ok, physics_worse, lead_da=0.0,
            dx_l2=0.5, dx_max=0.4, resid=0.0):
    return {"k0": k0, "bg": ({"rmse": bg_rmse}, _dev(0.0, resid)),
            "da": ({"rmse": da_rmse}, _dev(lead_da, resid)),
            "win": (_dev(0.04), _dev(0.04)),
            "rmse_delta_da_minus_bg": da_rmse - bg_rmse,
            "gate_da_vs_bg": (gate_ok, [] if gate_ok else ["...worse..."]),
            "physics_worse": physics_worse, "dx_l2": dx_l2, "dx_max_abs": dx_max}


def test_case_row_flags():
    row = _case_row(_result(2000, 0.20, 0.22, gate_ok=True, physics_worse=False))
    assert row["k0"] == 2000
    assert row["skill_improved"] is True and row["gate_pass"] is True
    assert row["physics_worse"] is False and row["state_large"] is False
    assert set(_COLS).issubset(row.keys())


def test_case_row_state_large_and_skill_regression():
    row = _case_row(_result(3000, 0.30, 0.22, gate_ok=False, physics_worse=True,
                            lead_da=0.3, dx_l2=40.0, dx_max=35.0))
    assert row["skill_improved"] is False        # da worse than bg
    assert row["gate_pass"] is False and row["physics_worse"] is True
    assert row["state_large"] is True            # dx_l2>3


def test_summarize_empty_is_report_only():
    s = summarize_multi([])
    assert s["n_valid"] == 0 and s["promotion"][0] == "REPORT_ONLY"


def test_summarize_all_pass_still_report_only_single_fixture():
    # every window passes -> beats-all True, but n_cases is pinned to 1 (one fixture),
    # so promotion stays REPORT_ONLY by design §11.
    rows = [_case_row(_result(k, 0.20, 0.22, gate_ok=True, physics_worse=False))
            for k in (1500, 2100, 2700, 3300)]
    s = summarize_multi(rows)
    assert s["gate_pass_rate"] == 1.0 and s["all_beat"] is True
    assert s["skill_improved_rate"] == 1.0 and s["physics_worse_rate"] == 0.0
    assert s["residual_clean"] is True
    verdict, reasons = s["promotion"]
    assert verdict == "REPORT_ONLY" and any("insufficient cases" in x for x in reasons)


def test_summarize_reports_physics_worse_rate_and_non_beat():
    # one window improves skill yet physics_worse -> gate fails there; rates reflect it.
    rows = [
        _case_row(_result(1500, 0.20, 0.22, gate_ok=True, physics_worse=False)),
        _case_row(_result(2100, 0.19, 0.22, gate_ok=False, physics_worse=True, lead_da=0.3)),
    ]
    s = summarize_multi(rows)
    assert s["gate_pass_rate"] == 0.5 and s["all_beat"] is False
    assert s["skill_improved_rate"] == 1.0        # both improved RMSE
    assert s["physics_worse_rate"] == 0.5         # but one worsened physics
    assert s["max_lead_diag_da"] == pytest.approx(0.3)
    assert s["promotion"][0] == "REPORT_ONLY"
    assert any("does not beat baseline in every window" in x for x in s["promotion"][1])


def test_summarize_dirty_residual_blocks_promotion():
    # even if skill passes everywhere, a dirty aggregate residual must block promotion
    # AND surface a reason (the code-leak detector fired) — not just report residual_clean.
    rows = [_case_row(_result(1500, 0.20, 0.22, gate_ok=True, physics_worse=False, resid=1e-6))]
    s = summarize_multi(rows)
    assert s["residual_clean"] is False and s["max_residual"] == pytest.approx(1e-6)
    verdict, reasons = s["promotion"]
    assert verdict == "REPORT_ONLY" and any("aggregate residual" in x for x in reasons)


def test_load_partial_ignores_stale_or_bad(tmp_path, monkeypatch):
    import tools.report_forecast_da_fullmodel_multi as mod
    import json
    p = tmp_path / "partial.json"
    monkeypatch.setattr(mod, "PARTIAL", p)
    # a bare-list (old schema) partial -> ignored, not crash
    p.write_text(json.dumps([{"k0": 1500}]))
    assert mod._load_partial() == {}
    # wrong config -> ignored
    p.write_text(json.dumps({"schema_version": mod._SCHEMA,
                             "config": {"stride": 999}, "rows": [{"k0": 1500}]}))
    assert mod._load_partial() == {}
    # matching schema + config -> loaded
    p.write_text(json.dumps({"schema_version": mod._SCHEMA, "config": mod._CONFIG,
                             "rows": [{"k0": 1500, "gate_pass": True}]}))
    assert set(mod._load_partial().keys()) == {1500}


@pytest.mark.jax
def test_multi_smoke_case_row_schema():
    from tools.report_forecast_da_fullmodel import build_a0
    r = build_a0(k0=2000, window=30, lead=60)
    row = _case_row(r)
    assert row["k0"] == 2000 and isinstance(row["gate_pass"], bool)
    assert row["resid_da"] < 1e-8                 # full-model audit clean
    s = summarize_multi([row])
    assert s["n_valid"] == 1 and s["promotion"][0] == "REPORT_ONLY"

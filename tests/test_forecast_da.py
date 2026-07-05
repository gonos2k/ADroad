"""Forecast-DA (state estimation) report: pure row/slice logic (fast) + a
jax-marked end-to-end smoke test of the assimilate->forecast cycle."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da import _slice_forc, _rows, _COLS  # noqa: E402


def _m(rmse, mae=0.1, ft=1.0):
    return {"rmse": rmse, "mae": mae, "freeze_thaw_accuracy": ft}


def test_slice_forc_slices_every_array():
    forc = {"Tair": list(range(10)), "obs": list(range(10, 20))}
    sl = _slice_forc(forc, 2, 5)
    assert sl["Tair"] == [2, 3, 4]
    assert sl["obs"] == [12, 13, 14]


def test_rows_structure_and_gate_labels():
    r = {"const": _m(0.9), "bg": (_m(0.22), 0.35), "da": (_m(0.20), 0.21),
         "gate_da_vs_bg": (True, []), "gate_da_vs_const": (True, []),
         "gate_bg_vs_const": (True, []), "degradation_bg": 0.63, "degradation_da": 0.95}
    rows = _rows(r)
    assert [x["model"] for x in rows] == ["constant_initial", "no_DA(background)", "DA(state)"]
    assert rows[0]["train_rmse"] == "" and rows[0]["degradation_ratio"] == ""   # baseline blanks
    assert rows[2]["gate_vs_bg"] == "PASS"                                      # DA beats background
    assert set(_COLS).issubset(rows[2].keys())


def test_rows_no_da_uses_own_gate_vs_const():
    # no-DA row must show ITS OWN gate vs const, not DA's (regression for the bug where
    # the background row reused gate_da_vs_const).
    r = {"const": _m(0.9), "bg": (_m(0.22), 0.35), "da": (_m(0.20), 0.21),
         "gate_da_vs_bg": (True, []), "gate_da_vs_const": (True, []),
         "gate_bg_vs_const": (False, ["worse than const"]),
         "degradation_bg": 0.63, "degradation_da": 0.95}
    rows = _rows(r)
    assert rows[1]["gate_vs_const"].startswith("FAIL")   # background's own verdict
    assert rows[2]["gate_vs_const"] == "PASS"            # DA's verdict is independent


def test_rows_marks_da_failure_when_worse():
    r = {"const": _m(0.9), "bg": (_m(0.20), 0.35), "da": (_m(0.30), 0.21),
         "gate_da_vs_bg": (False, ["forecast RMSE 0.3000 worse than baseline 0.2000"]),
         "gate_da_vs_const": (True, []), "gate_bg_vs_const": (True, []),
         "degradation_bg": 0.6, "degradation_da": 1.4}
    rows = _rows(r)
    assert rows[2]["gate_vs_bg"].startswith("FAIL")      # honest: DA worse than no-DA -> FAIL


def _res(k0, da, bg, beats):
    return {"k0": k0, "da": (_m(da), 0.2), "bg": (_m(bg), 0.3),
            "rmse_delta_da_minus_bg": da - bg, "degradation_da": 1.0, "dx_l2": 0.5,
            "gate_da_vs_bg": (beats, [] if beats else ["worse"])}


def test_multi_summarize_reportonly_when_not_all_beat():
    from tools.report_forecast_da_multi import summarize
    # enough cases (3) but DA loses one window -> promotion stays REPORT_ONLY
    rows, all_beat, (verdict, reasons) = summarize(
        [_res(1500, 0.6, 0.5, False), _res(2100, 0.4, 0.5, True), _res(2700, 0.45, 0.46, True)])
    assert len(rows) == 3 and all_beat is False
    assert verdict == "REPORT_ONLY" and any("every window" in r for r in reasons)


def test_multi_summarize_promote_when_all_beat_enough_cases():
    from tools.report_forecast_da_multi import summarize
    rows, all_beat, (verdict, _r) = summarize([_res(k, 0.4, 0.5, True) for k in (1500, 2100, 2700)])
    assert all_beat is True and verdict == "PROMOTE"


@pytest.mark.jax
def test_forecast_da_smoke():
    from tools.report_forecast_da import build
    r = build(k0=2000, window=30, lead=30)
    for key in ("da", "bg", "const"):
        pass
    assert r["valid_win"] >= 3 and r["valid_lead"] >= 3
    assert len(r["dx"]) == 4
    import math
    assert math.isfinite(r["da"][0]["rmse"]) and math.isfinite(r["bg"][0]["rmse"])
    assert isinstance(r["gate_da_vs_bg"][0], bool)
    assert math.isfinite(r["rmse_delta_da_minus_bg"])

"""Design A0 full-model forecast DA: pure row logic (fast) + jax-marked smoke."""
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da_fullmodel import _rows, _COLS  # noqa: E402


def _m(rmse, mae=0.1, ft=1.0):
    return {"rmse": rmse, "mae": mae, "freeze_thaw_accuracy": ft}


def _dev(res=0.0, om=0, of=0):
    return {"max_primary_residual": res, "over_melt_count": om, "overflow_count": of}


def test_rows_structure_and_pass_label():
    r = {"const": _m(0.9), "bg": (_m(0.22), _dev()), "da": (_m(0.20), _dev()),
         "gate_da_vs_bg": (True, [])}
    rows = _rows(r)
    assert [x["model"] for x in rows] == ["constant_initial", "no_DA(background)", "DA(state, full)"]
    assert rows[2]["gate_vs_bg"] == "PASS"
    assert rows[0]["max_primary_residual"] == ""          # constant_initial has no deviation
    assert set(_COLS).issubset(rows[1].keys())


def test_rows_fail_label_when_worse():
    r = {"const": _m(0.9), "bg": (_m(0.20), _dev()), "da": (_m(0.30), _dev()),
         "gate_da_vs_bg": (False, ["forecast RMSE 0.3000 worse than baseline 0.2000"])}
    assert _rows(r)[2]["gate_vs_bg"].startswith("FAIL")


@pytest.mark.jax
def test_a0_smoke():
    from tools.report_forecast_da_fullmodel import build_a0
    r = build_a0(k0=2000, window=30, lead=60)
    assert math.isfinite(r["bg"][0]["rmse"]) and math.isfinite(r["da"][0]["rmse"])
    assert isinstance(r["gate_da_vs_bg"][0], bool) and isinstance(r["physics_worse"], bool)
    assert len(r["dx"]) == 4 and math.isfinite(r["rmse_delta_da_minus_bg"])
    # full model evolves storages -> mass audit residual must stay ~0 (code-leak gate)
    assert r["da"][1]["max_primary_residual"] < 1e-8
    assert r["bg"][1]["max_primary_residual"] < 1e-8

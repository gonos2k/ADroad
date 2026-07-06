"""Design A0 full-model forecast DA: pure row logic (fast) + jax-marked smoke."""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da_fullmodel import (  # noqa: E402
    _rows, _COLS, _inject_dx_state, _forecast_kwargs,
)


def _fake_objs(n=50):
    """Minimal objs tuple (mi, mo, phy, g, s, a, coup, st, cpm, _) for unit-testing the
    forecast-kwargs / dx-injection contract without building the real model."""
    ns = types.SimpleNamespace
    arr = lambda v: np.full(n, v, float)
    mi = ns(TSurfObs=arr(0.0), Tair=arr(-1.0), VZ=arr(2.0), Rhz=arr(80.0), SW=arr(0.0),
            LW=arr(300.0), PrecPhase=arr(0.0), prec=arr(0.0),
            time=[ns(hour=i % 24) for i in range(n)])
    g = ns(Tmp=np.array([-1.0, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
           TmpNw=np.array([-1.0, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
           WCont=np.ones(8), CC=np.ones(8), ZDpth=np.ones(8), DyK=np.ones(8), DyC=np.ones(8),
           Albedo=0.1)
    s = ns(SrfWatmms=0.0, SrfSnowmms=0.0, SrfIcemms=0.0, SrfIce2mms=0.0, SrfDepmms=0.0,
           TsurfAve=-0.25, Q2Melt=0.0, T4Melt=0.0, WearSurf=0.0, VeryCold=False)
    a = ns(BLCond=1.0, SnowType=0)
    st = ns(NLayers=6, DTSecs=30.0, MaxPormms=1.0, Tph=0.0, NightOn=20, NightOff=6,
            CalmLimDay=0.5, CalmLimNgt=0.5, TrfFricDay=0.0, TrfFricNgt=0.0)
    phy = ns(Poro1=0.3, Poro2=0.3, vsh1=1e6, vsh2=1e6, Emiss=0.95, SB_const=5.67e-8,
             VK_Const=0.4, logUstar=1.0, logCond=1.0, logMom=1.0, logHeat=1.0, ZRefT=2.0,
             Grav=9.81, LVap=2.5e6, LFus=3.34e5, MaxPormms=1.0)
    cpm = ns(WetSnowFrozen=False)
    coup = ns(LastTsurfObs=-9999.0)
    return (mi, None, phy, g, s, a, coup, st, cpm, None)


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


def test_forecast_kwargs_disables_obs_insertion():
    # the no-future-obs-leakage contract: InitLenI=-1, sentinel TSurfObs, coupling off.
    kw = _forecast_kwargs(_fake_objs(50), k0=10, span=20, dx=np.array([1.0, 1.0, 1.0, 1.0]))
    assert kw["InitLenI"] == -1
    assert bool(np.all(kw["TSurfObs"] == -9999.0))
    assert kw["inCouplingPhase"] is False
    assert kw["return_ledger"] is True
    assert kw["n_steps"] == 20 and len(kw["Tair"]) == 20


def test_forecast_kwargs_range_and_arg_guards():
    with pytest.raises(RuntimeError):
        _forecast_kwargs(_fake_objs(30), k0=25, span=20)          # 45 > 30 (out of data)
    with pytest.raises(RuntimeError):
        _forecast_kwargs(_fake_objs(30), k0=-1, span=10)          # k0 < 0
    with pytest.raises(RuntimeError):
        _forecast_kwargs(_fake_objs(30), k0=0, span=0)            # span <= 0
    with pytest.raises(RuntimeError):
        _forecast_kwargs(_fake_objs(30), k0=1.5, span=10)         # non-integer k0


def test_inject_dx_rejects_bad_shape_or_nonfinite():
    objs = _fake_objs(30)
    with pytest.raises(ValueError):
        _inject_dx_state(objs, np.array([1.0, 1.0, 1.0]))         # shape (3,)
    with pytest.raises(ValueError):
        _inject_dx_state(objs, np.array([1.0, float("nan"), 1.0, 1.0]))   # non-finite
    with pytest.raises(ValueError):
        _inject_dx_state(objs, np.array([True, True, True, True]))         # bool dtype
    with pytest.raises(ValueError):
        _inject_dx_state(objs, np.array(["1", "1", "1", "1"]))            # string dtype


def test_inject_dx_syncs_tsurfave_and_isolates_state():
    objs = _fake_objs(50)
    g, s = objs[3], objs[4]
    orig_tmp = g.Tmp.copy()
    Tmp0, TmpNw0, surf0 = _inject_dx_state(objs, np.array([2.0, 2.0, 2.0, 2.0]))
    assert surf0.TsurfAve == pytest.approx((Tmp0[1] + Tmp0[2]) / 2.0)   # synced
    assert not np.shares_memory(Tmp0, g.Tmp)                            # isolated
    assert np.allclose(g.Tmp, orig_tmp)                                # objs not mutated
    assert Tmp0[1] == pytest.approx(orig_tmp[1] + 2.0)                 # dx applied to 1:5
    # background (dx=None) keeps the physical Surf.TsurfAve
    _, _, surf_bg = _inject_dx_state(objs, None)
    assert surf_bg.TsurfAve == s.TsurfAve


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


@pytest.mark.jax
def test_a0_storage_active_smoke():
    # k0=3800 window covers active storage (Ice near step 4312): diagnostics fire but the
    # mass-audit residual must stay within the P0 gate tolerance (code-leak detector).
    from tools.report_forecast_da_fullmodel import build_a0
    r = build_a0(k0=3800, window=120, lead=480)
    assert r["valid_lead"] >= 3
    assert math.isfinite(r["da"][0]["rmse"]) and isinstance(r["gate_da_vs_bg"][0], bool)
    assert "diagnostic_steps_rate" in r["da"][1]
    assert r["da"][1]["max_primary_residual"] < 1e-9     # within P0 gate tolerance
    assert r["bg"][1]["max_primary_residual"] < 1e-9

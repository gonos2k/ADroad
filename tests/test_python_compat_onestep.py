"""G1a python_compat: droad dry one-step vs RoadSurf-Python BalanceModelOneStep.

Runs on the real initialized state at step 0 (dry: no precip / storage), so the
reference melting/storage paths are no-ops and the comparison isolates the dry
thermal + boundary-layer + radiation path.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.model import balance_one_step_dry  # noqa: E402
from droad.boundary import calc_blc_and_le  # noqa: E402


def _phy_dict(phy):
    return {
        "Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
        "Emiss": phy.Emiss, "SB_const": phy.SB_const,
        "VK": phy.VK_Const, "logUstar": phy.logUstar, "logCond": phy.logCond,
        "logMom": phy.logMom, "logHeat": phy.logHeat,
        "ZRefT": phy.ZRefT, "Grav": phy.Grav, "LVap": phy.LVap, "LFus": phy.LFus,
    }


def _day_dict(s):
    return {
        "NightOn": s.NightOn, "NightOff": s.NightOff,
        "CalmLimDay": s.CalmLimDay, "CalmLimNgt": s.CalmLimNgt,
        "TrfFricDay": s.TrfFricDay, "TrfFricNgt": s.TrfFricNgt,
    }


@pytest.fixture(scope="module")
def prepared():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    (modelInput, modelOutput, phy, ground, surf, atm, coupling, settings,
     condParam, localParam) = objs
    # set current values for step 0 (as the reference loop does)
    m["InputOutput"].SetCurrentValues(0, modelInput, atm, settings, surf, coupling, ground)
    return m, objs


def test_no_sky_view_correction(prepared):
    # this dry parity assumes ModRadiation is inactive (raw SW/LW used)
    _, objs = prepared
    localParam = objs[9]
    assert not (-0.01 < localParam.sky_view < 1.0)


def test_boundary_layer_matches_reference(prepared):
    m, objs = prepared
    _, _, phy, ground, surf, atm, _, settings, _, _ = objs
    # apply day/night VZ clamp exactly like the reference does before BLC
    from droad.model import set_day_dependent
    hour = objs[0].time[0].hour
    _, vzc, _ = set_day_dependent(hour, atm.VZ, _day_dict(settings))

    ac, sc = copy.deepcopy(atm), copy.deepcopy(surf)
    ac.VZ = vzc
    m["BoundaryLayer"].CalcBLCondAndLE(sc, settings.DTSecs, sc.SrfWatmms, phy, ac)  # noqa

    BLCond, LE, Evap = calc_blc_and_le(
        surf.TsurfAve, atm.Tair, vzc, atm.Rhz, atm.BLCond, surf.SrfWatmms,
        settings.DTSecs, _phy_dict(phy))
    assert BLCond == pytest.approx(ac.BLCond, abs=1e-9, rel=0)
    assert LE == pytest.approx(ac.LE_Flux, abs=1e-9, rel=0)
    assert Evap == pytest.approx(sc.EvapmmTS, abs=1e-9, rel=0)


def test_dry_one_step_matches_reference(prepared):
    m, objs = prepared
    modelInput, _, phy, ground, surf, atm, coupling, settings, condParam, _ = objs

    # reference: full one-step on clones
    gc, sc, ac = copy.deepcopy(ground), copy.deepcopy(surf), copy.deepcopy(atm)
    m["BalanceModel"].BalanceModelOneStep(
        modelInput.SW[0], modelInput.LW[0], phy, gc, sc, ac, settings,
        coupling, modelInput, 0, condParam)
    # guard: step 0 must be dry (no storage) for this comparison to be valid
    assert (sc.SrfSnowmms, sc.SrfIcemms, sc.SrfIce2mms, sc.SrfDepmms) == (0.0, 0.0, 0.0, 0.0)

    out = balance_one_step_dry(
        Tmp=np.array(ground.Tmp, float), TmpNw=np.array(ground.TmpNw, float),
        WCont=np.array(ground.WCont, float), CC=np.array(ground.CC, float),
        ZDpth=np.array(ground.ZDpth, float), DyK=np.array(ground.DyK, float),
        DyC=np.array(ground.DyC, float), Albedo=ground.Albedo,
        TsurfAve=surf.TsurfAve, SrfWat=surf.SrfWatmms,
        Tair=atm.Tair, VZ=atm.VZ, Rhz=atm.Rhz, BLCond_init=atm.BLCond,
        SW=modelInput.SW[0], LW=modelInput.LW[0], hour=modelInput.time[0].hour,
        NLayers=settings.NLayers, DTSecs=settings.DTSecs,
        phy=_phy_dict(phy), day=_day_dict(settings))

    assert np.allclose(out["Tmp"], np.array(gc.TmpNw, float), atol=1e-9, rtol=0)
    assert out["TsurfAve"] == pytest.approx(sc.TsurfAve, abs=1e-9)
    assert out["BLCond"] == pytest.approx(ac.BLCond, abs=1e-9)


def test_blc_v0_v1_agree(prepared):
    # fixed-unroll (v0) and early-stop (v1) converge to the same BLCond
    m, objs = prepared
    _, _, phy, _, surf, atm, _, settings, _, _ = objs
    p = _phy_dict(phy)
    v1 = calc_blc_and_le(surf.TsurfAve, atm.Tair, atm.VZ, atm.Rhz, atm.BLCond,
                         surf.SrfWatmms, settings.DTSecs, p, early_stop=True)
    v0 = calc_blc_and_le(surf.TsurfAve, atm.Tair, atm.VZ, atm.Rhz, atm.BLCond,
                         surf.SrfWatmms, settings.DTSecs, p, early_stop=False)
    # they agree to within the solver convergence limit (0.001), not to machine eps
    assert v0[0] == pytest.approx(v1[0], abs=1e-3)

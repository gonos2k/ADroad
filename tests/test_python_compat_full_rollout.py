"""G2 capstone: free-running full model rollout vs RoadSurf-Python no-coupling.

droad carries its own full state (temps + 5 storages + flags + BLCond + albedo)
and must track the reference no-coupling run across the whole trajectory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.driver import full_rollout  # noqa: E402
from droad.storage import Surf  # noqa: E402


def _phy(phy):
    return {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
            "Emiss": phy.Emiss, "SB_const": phy.SB_const, "VK": phy.VK_Const,
            "logUstar": phy.logUstar, "logCond": phy.logCond, "logMom": phy.logMom,
            "logHeat": phy.logHeat, "ZRefT": phy.ZRefT, "Grav": phy.Grav,
            "LVap": phy.LVap, "LFus": phy.LFus}


def _day(s):
    return {"NightOn": s.NightOn, "NightOff": s.NightOff, "CalmLimDay": s.CalmLimDay,
            "CalmLimNgt": s.CalmLimNgt, "TrfFricDay": s.TrfFricDay, "TrfFricNgt": s.TrfFricNgt}


def _cp(cp):
    return {k: getattr(cp, k) for k in dir(cp) if not k.startswith("_")}


def _reference_trajectory(m, objs, n):
    mi, mo, phy, ground, surf, atm, coupling, settings, condParam, _ = objs
    out = {k: np.empty(n) for k in ("Tsurf", "Snow", "Water", "Ice", "Ice2", "Dep")}
    for i in range(n):
        m["InputOutput"].SetCurrentValues(i, mi, atm, settings, surf, coupling, ground)
        m["Storage"].PrecipitationToStorage(settings, condParam, mi.PrecPhase[i], atm, surf)
        m["BalanceModel"].BalanceModelOneStep(mi.SW[i], mi.LW[i], phy, ground, surf,
                                              atm, settings, coupling, mi, i, condParam)
        wf = m["WearingFactors"].WearingFactors()
        m["Cond"].WearFactors(condParam, settings.Tph, surf, wf)
        m["Cond"].RoadCond(phy.MaxPormms, surf, atm, settings, condParam, wf)
        ground.Albedo = m["Cond"].CalcAlbedo(surf, condParam)
        out["Tsurf"][i] = surf.TsurfAve
        out["Snow"][i] = surf.SrfSnowmms
        out["Water"][i] = surf.SrfWatmms
        out["Ice"][i] = surf.SrfIcemms
        out["Ice2"][i] = surf.SrfIce2mms
        out["Dep"][i] = surf.SrfDepmms
    return out


def test_full_rollout_matches_reference():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    settings = objs[7]
    n = settings.SimLen - 1
    ref = _reference_trajectory(m, objs, n)

    m2, o2 = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = o2
    hours = np.array([t.hour for t in mi.time[:n]], float)
    prec_in = np.array(mi.prec, float) / 3600.0 * st.DTSecs
    surf0 = Surf(SrfWat=s.SrfWatmms, SrfSnow=s.SrfSnowmms, SrfIce=s.SrfIcemms,
                 SrfIce2=s.SrfIce2mms, SrfDep=s.SrfDepmms, TsurfAve=s.TsurfAve,
                 Q2Melt=s.Q2Melt, T4Melt=s.T4Melt, WearSurf=s.WearSurf,
                 SnowType=a.SnowType, WetSnowFrozen=cpm.WetSnowFrozen, VeryCold=s.VeryCold)

    got = full_rollout(
        Tair=np.array(mi.Tair, float), VZ=np.array(mi.VZ, float), Rhz=np.array(mi.Rhz, float),
        SW=np.array(mi.SW, float), LW=np.array(mi.LW, float),
        TSurfObs=np.array(mi.TSurfObs, float), hours=hours,
        prec_phase=np.array(mi.PrecPhase, float), prec_in_tstep=prec_in,
        Tmp0=g.Tmp, TmpNw0=g.TmpNw, WCont=np.array(g.WCont, float),
        CC=np.array(g.CC, float), ZDpth=np.array(g.ZDpth, float),
        DyK=np.array(g.DyK, float), DyC=np.array(g.DyC, float),
        surf0=surf0, Albedo0=g.Albedo, BLCond0=a.BLCond,
        NLayers=st.NLayers, DTSecs=st.DTSecs, MaxPormms=phy.MaxPormms, Tph=st.Tph,
        InitLenI=st.InitLenI, phy=_phy(phy), day=_day(st), cp=_cp(cpm), n_steps=n,
        TsurfObsLast=coup.LastTsurfObs)

    tsurf_rmse = float(np.sqrt(np.mean((got["Tsurf"] - ref["Tsurf"]) ** 2)))
    snow_mae = float(np.mean(np.abs(got["Snow"] - ref["Snow"])))
    water_mae = float(np.mean(np.abs(got["Water"] - ref["Water"])))
    ice_mae = float(np.mean(np.abs(got["Ice"] - ref["Ice"])))

    assert tsurf_rmse <= 1e-2, f"Tsurf RMSE={tsurf_rmse:.3e}"
    assert snow_mae <= 1e-3 and water_mae <= 1e-3 and ice_mae <= 1e-3, \
        f"storage MAE snow={snow_mae:.3e} water={water_mae:.3e} ice={ice_mae:.3e}"

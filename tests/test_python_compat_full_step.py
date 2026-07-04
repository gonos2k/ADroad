"""G2 python_compat: full-model step vs RoadSurf-Python over the whole run.

Teacher-forced: each step both sides start from the reference's post-
SetCurrentValues state; droad step_full is compared to the reference
roadModelOneStep result (BalanceModelOneStep + WearFactors + RoadCond + Albedo).
This exercises precipitation & phase-change branches on the real trajectory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.model import step_full  # noqa: E402
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


def _cp(condParam):
    return {k: getattr(condParam, k) for k in dir(condParam) if not k.startswith("_")}


def test_full_step_matches_reference_all_steps():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    mi, mo, phy, ground, surf, atm, coupling, settings, condParam, localParam = objs

    phy_d, day_d = _phy(phy), _day(phy) if False else _day(settings)
    n = settings.SimLen - 1
    max_dT = max_dstore = 0.0

    for i in range(n):
        m["InputOutput"].SetCurrentValues(i, mi, atm, settings, surf, coupling, ground)

        pre = Surf(
            SrfWat=surf.SrfWatmms, SrfSnow=surf.SrfSnowmms, SrfIce=surf.SrfIcemms,
            SrfIce2=surf.SrfIce2mms, SrfDep=surf.SrfDepmms, TsurfAve=surf.TsurfAve,
            EvapmmTS=surf.EvapmmTS, Q2Melt=surf.Q2Melt, T4Melt=surf.T4Melt,
            WearSurf=surf.WearSurf, SnowType=atm.SnowType,
            WetSnowFrozen=condParam.WetSnowFrozen, VeryCold=surf.VeryCold)

        out = step_full(
            Tmp=np.array(ground.Tmp, float), TmpNw=np.array(ground.TmpNw, float),
            WCont=np.array(ground.WCont, float), CC=np.array(ground.CC, float),
            ZDpth=np.array(ground.ZDpth, float), DyK=np.array(ground.DyK, float),
            DyC=np.array(ground.DyC, float), surf=pre, Albedo=ground.Albedo,
            BLCond=atm.BLCond, Tair=atm.Tair, VZ=atm.VZ, Rhz=atm.Rhz,
            SW=mi.SW[i], LW=mi.LW[i], hour=mi.time[i].hour,
            prec_phase=mi.PrecPhase[i], prec_in_tstep=atm.PrecInTStep,
            inCouplingPhase=coupling.inCouplingPhase, TsurfObsLast=coupling.LastTsurfObs,
            NLayers=settings.NLayers, DTSecs=settings.DTSecs, MaxPormms=phy.MaxPormms,
            Tph=settings.Tph, phy=phy_d, day=day_d, cp=_cp(condParam))

        # advance reference exactly like roadModelOneStep (no sky-view)
        m["Storage"].PrecipitationToStorage(settings, condParam, mi.PrecPhase[i], atm, surf)
        m["BalanceModel"].BalanceModelOneStep(mi.SW[i], mi.LW[i], phy, ground, surf,
                                              atm, settings, coupling, mi, i, condParam)
        wf = m["WearingFactors"].WearingFactors()
        m["Cond"].WearFactors(condParam, settings.Tph, surf, wf)
        m["Cond"].RoadCond(phy.MaxPormms, surf, atm, settings, condParam, wf)
        ground.Albedo = m["Cond"].CalcAlbedo(surf, condParam)

        assert np.allclose(out["TmpNw"], np.array(ground.TmpNw, float), atol=1e-9, rtol=0), f"TmpNw step {i}"
        assert out["TsurfAve"] == pytest.approx(surf.TsurfAve, abs=1e-9), f"Tsurf step {i}"
        assert out["Albedo"] == pytest.approx(ground.Albedo, abs=1e-12), f"Albedo step {i}"
        assert out["surf"].SrfWat == pytest.approx(surf.SrfWatmms, abs=1e-9), f"Wat step {i}"
        assert out["surf"].SrfSnow == pytest.approx(surf.SrfSnowmms, abs=1e-9), f"Snow step {i}"
        assert out["surf"].SrfIce == pytest.approx(surf.SrfIcemms, abs=1e-9), f"Ice step {i}"
        assert out["surf"].SrfIce2 == pytest.approx(surf.SrfIce2mms, abs=1e-9), f"Ice2 step {i}"
        assert out["surf"].SrfDep == pytest.approx(surf.SrfDepmms, abs=1e-9), f"Dep step {i}"

        max_dT = max(max_dT, abs(out["TsurfAve"] - surf.TsurfAve))
        max_dstore = max(max_dstore, abs(out["surf"].SrfSnow - surf.SrfSnowmms))

    assert max_dT < 1e-9 and max_dstore < 1e-9

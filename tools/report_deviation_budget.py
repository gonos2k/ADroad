#!/usr/bin/env python3
"""Generate the baseline deviation budget from the no-coupling reference rollout.

Runs droad.driver.full_rollout(return_ledger=True) over the example case and
aggregates its audit trail (droad.deviation) into reports/. This is the common
substrate for forecast-skill-gate and diagnostics-aware DA evaluation:

    python3 tools/report_deviation_budget.py

Writes reports/deviation_budget_baseline.md and .csv.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402
from droad.driver import full_rollout  # noqa: E402
from droad.storage import Surf  # noqa: E402
from droad.deviation import deviation_budget, budget_to_markdown, budget_to_csv  # noqa: E402


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


def run_baseline():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    n = st.SimLen - 1
    hours = np.array([t.hour for t in mi.time[:n]], float)
    prec_in = np.array(mi.prec, float) / 3600.0 * st.DTSecs
    surf0 = Surf(SrfWat=s.SrfWatmms, SrfSnow=s.SrfSnowmms, SrfIce=s.SrfIcemms,
                 SrfIce2=s.SrfIce2mms, SrfDep=s.SrfDepmms, TsurfAve=s.TsurfAve,
                 Q2Melt=s.Q2Melt, T4Melt=s.T4Melt, WearSurf=s.WearSurf,
                 SnowType=a.SnowType, WetSnowFrozen=cpm.WetSnowFrozen, VeryCold=s.VeryCold)
    out = full_rollout(
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
        TsurfObsLast=coup.LastTsurfObs, return_ledger=True)
    return deviation_budget(out, case_id="no_coupling_baseline")


def main():
    summary = run_baseline()
    outdir = REPO / "reports"
    outdir.mkdir(exist_ok=True)
    (outdir / "deviation_budget_baseline.md").write_text(
        budget_to_markdown([summary], "Deviation Budget — baseline (no-coupling)"), encoding="utf-8")
    (outdir / "deviation_budget_baseline.csv").write_text(
        budget_to_csv([summary]), encoding="utf-8")
    print("wrote reports/deviation_budget_baseline.{md,csv}")
    for k in ("n_steps", "max_primary_residual", "n_diagnostics_total",
              "diagnostic_steps_rate", "over_melt_count", "overflow_count",
              "negative_pre_clamp_count", "max_storage_jump"):
        print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    main()

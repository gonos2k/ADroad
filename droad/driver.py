"""Free-running dry rollout (M1, G1b).

Mirrors RoadSurf-Python's driver for the dry path: per step it applies the
observation forcing (SetCurrentValues: air temp into layer 0, and — during the
initialization window — observed surface temp into layers 1-2) and then advances
one dry thermal step. No storage/precipitation (dry).

State carried across steps: Tmp, TmpNw (kept separate — obs forcing writes Tmp
layers 1-2 but heat capacity reads the un-forced TmpNw, exactly as reference),
plus TsurfAve, BLCond, Albedo.
"""

from __future__ import annotations

import numpy as np

from .model import balance_one_step_dry, step_full
from .storage import Surf


def dry_rollout(*, Tair, VZ, Rhz, SW, LW, TSurfObs, hours,
                Tmp0, TmpNw0, WCont, CC, ZDpth, DyK, DyC,
                Albedo, BLCond0, TsurfAve0, NLayers, DTSecs, InitLenI,
                phy, day, n_steps, early_stop=True):
    Tmp = np.array(Tmp0, float).copy()
    TmpNw = np.array(TmpNw0, float).copy()
    BLCond, TsurfAve = BLCond0, TsurfAve0
    tsurf = np.empty(n_steps)

    for i in range(n_steps):
        Tmp[0] = Tair[i]                                   # SetCurrentValues
        if i <= InitLenI and TSurfObs[i] > -100.0:         # obs forcing (coupling off)
            Tmp[1] = TSurfObs[i]
            Tmp[2] = TSurfObs[i]
            TsurfAve = (Tmp[1] + Tmp[2]) / 2.0

        out = balance_one_step_dry(
            Tmp=Tmp, TmpNw=TmpNw, WCont=WCont, CC=CC, ZDpth=ZDpth, DyK=DyK, DyC=DyC,
            Albedo=Albedo, TsurfAve=TsurfAve, SrfWat=0.0,
            Tair=Tair[i], VZ=VZ[i], Rhz=Rhz[i], BLCond_init=BLCond,
            SW=SW[i], LW=LW[i], hour=hours[i],
            NLayers=NLayers, DTSecs=DTSecs, phy=phy, day=day, early_stop=early_stop)

        Tmp = out["Tmp"].copy()          # reference: Tmp = TmpNw.copy() at step end
        TmpNw = out["Tmp"].copy()
        TsurfAve = out["TsurfAve"]
        BLCond = out["BLCond"]
        tsurf[i] = TsurfAve

    return tsurf


def full_rollout(*, Tair, VZ, Rhz, SW, LW, TSurfObs, hours, prec_phase, prec_in_tstep,
                 Tmp0, TmpNw0, WCont, CC, ZDpth, DyK, DyC, surf0: Surf, Albedo0, BLCond0,
                 NLayers, DTSecs, MaxPormms, Tph, InitLenI, phy, day, cp, n_steps,
                 inCouplingPhase=False, TsurfObsLast=-9999.0, return_ledger=False):
    """Free-running full model (dry + storage/phase-change). Returns a dict of
    per-step trajectories (Tsurf + 5 storages). With return_ledger=True the dict
    also carries, per step:
      out["ledger"]        = the merged full-step StorageLedger (precip + condition)
      out["ledger_detail"] = (prec_ledger, cond_ledger) for drilling in
      out["diagnostics"]   = tuple of feasibility flags (over-melt, overflow, ...)
    so mass accounting and phase diagnostics can be inspected post-hoc."""
    Tmp = np.array(Tmp0, float).copy()
    TmpNw = np.array(TmpNw0, float).copy()
    surf, Albedo, BLCond = surf0, Albedo0, BLCond0

    out = {k: np.empty(n_steps) for k in
           ("Tsurf", "Snow", "Water", "Ice", "Ice2", "Dep")}
    if return_ledger:
        out["ledger"] = []          # merged full-step ledger per step
        out["ledger_detail"] = []   # (prec, cond) per step
        out["diagnostics"] = []     # feasibility flags per step

    for i in range(n_steps):
        Tmp[0] = Tair[i]                                   # SetCurrentValues
        tsa = surf.TsurfAve
        if i <= InitLenI and TSurfObs[i] > -100.0:
            Tmp[1] = TSurfObs[i]
            Tmp[2] = TSurfObs[i]
            tsa = (Tmp[1] + Tmp[2]) / 2.0
        surf = replace_tsurf(surf, tsa)

        r = step_full(
            Tmp=Tmp, TmpNw=TmpNw, WCont=WCont, CC=CC, ZDpth=ZDpth, DyK=DyK, DyC=DyC,
            surf=surf, Albedo=Albedo, BLCond=BLCond,
            Tair=Tair[i], VZ=VZ[i], Rhz=Rhz[i], SW=SW[i], LW=LW[i], hour=hours[i],
            prec_phase=prec_phase[i], prec_in_tstep=prec_in_tstep[i],
            inCouplingPhase=inCouplingPhase, TsurfObsLast=TsurfObsLast,
            NLayers=NLayers, DTSecs=DTSecs, MaxPormms=MaxPormms, Tph=Tph,
            phy=phy, day=day, cp=cp)

        Tmp = r["TmpNw"].copy()
        TmpNw = r["TmpNw"].copy()
        surf, Albedo, BLCond = r["surf"], r["Albedo"], r["BLCond"]
        out["Tsurf"][i] = r["TsurfAve"]
        out["Snow"][i] = surf.SrfSnow
        out["Water"][i] = surf.SrfWat
        out["Ice"][i] = surf.SrfIce
        out["Ice2"][i] = surf.SrfIce2
        out["Dep"][i] = surf.SrfDep
        if return_ledger:
            out["ledger"].append(r["step_ledger"])
            out["ledger_detail"].append((r["prec_ledger"], r["cond_ledger"]))
            out["diagnostics"].append(r["diagnostics"])

    return out


def replace_tsurf(surf: Surf, tsurf):
    from dataclasses import replace
    return replace(surf, TsurfAve=tsurf)

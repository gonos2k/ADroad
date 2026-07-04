"""Assembled dry one-step (M1). Mirrors RoadSurf-Python BalanceModelOneStep for
the no-precip / no-storage case: day/night traffic -> BLC -> net radiation ->
ground heat capacity/conduction -> explicit profile step. Melting is a no-op
when there is no snow/ice, so it is omitted here (dry).
"""

from __future__ import annotations

from dataclasses import replace

from .boundary import calc_blc_and_le
from .radiation import calc_rnet
from .thermal import calc_hcap_hcond, calc_cap_cond, calc_profile, calc_hstor
from .ledger import merge_ledgers
from .storage import (
    calc_prec_type, precipitation_to_storage, wear_factors, melting, calc_albedo,
)
from .roadcond import road_cond


def set_day_dependent(hour, VZ, day):
    """Traffic friction + minimum wind (eq: SetDayDependendVariables).
    Returns (TrfFric, VZ_clamped, CalmLim)."""
    if hour >= day["NightOn"] or hour <= day["NightOff"]:
        CalmLim, TrfFric = day["CalmLimNgt"], day["TrfFricNgt"]
    else:
        CalmLim, TrfFric = day["CalmLimDay"], day["TrfFricDay"]
    return TrfFric, (VZ if VZ >= CalmLim else CalmLim), CalmLim


def balance_one_step_dry(*, Tmp, TmpNw, WCont, CC, ZDpth, DyK, DyC,
                         Albedo, TsurfAve, SrfWat, Tair, VZ, Rhz, BLCond_init,
                         SW, LW, hour, NLayers, DTSecs, phy, day, early_stop=True):
    """One dry thermal step. Returns a dict of the new state + fluxes."""
    TrfFric, VZc, _ = set_day_dependent(hour, VZ, day)
    BLCond, LE, Evap = calc_blc_and_le(TsurfAve, Tair, VZc, Rhz, BLCond_init,
                                       SrfWat, DTSecs, phy, early_stop=early_stop)
    RNet = calc_rnet(phy["Emiss"], phy["SB_const"], TsurfAve, Albedo, SW, LW)
    VSH, HS, GCond = calc_hcap_hcond(TmpNw, WCont, CC, ZDpth, NLayers, DTSecs, phy, BLCond)
    condDZ, capDZ = calc_cap_cond(CC, DyK, DyC, VSH, NLayers)
    TmpNw_new, GroundFlux = calc_profile(Tmp, condDZ, capDZ, NLayers, DTSecs,
                                         TrfFric, BLCond, RNet, LE)
    return {
        "Tmp": TmpNw_new,
        "TsurfAve": (TmpNw_new[1] + TmpNw_new[2]) / 2.0,
        "BLCond": BLCond, "LE": LE, "Evap": Evap, "RNet": RNet,
        "GroundFlux": GroundFlux, "TrfFric": TrfFric, "VZ": VZc,
    }


def step_full(*, Tmp, TmpNw, WCont, CC, ZDpth, DyK, DyC, surf, Albedo, BLCond,
              Tair, VZ, Rhz, SW, LW, hour, prec_phase, prec_in_tstep,
              inCouplingPhase, TsurfObsLast, NLayers, DTSecs, MaxPormms, Tph,
              phy, day, cp):
    """Full model step (precip -> balance+melting -> wear -> RoadCond -> albedo).
    Mirrors RoadSurf-Python roadModelOneStep for the no-sky-view case.
    `surf` is a storage.Surf; returns dict with new TmpNw, TsurfAve, BLCond,
    Albedo, and the updated Surf."""
    # 1) precipitation -> storage (via the ledgered contract)
    pt = calc_prec_type(prec_in_tstep, prec_phase, Tair, Rhz, DTSecs, cp)
    SrfWat_new, SrfSnow_new, prec_ledger = precipitation_to_storage(
        surf.SrfWat, surf.SrfSnow, surf.SrfIce, surf.SrfDep, pt)
    surf = replace(surf, SrfWat=SrfWat_new, SrfSnow=SrfSnow_new)

    # 2) balance (day/night -> BLC -> net radiation -> conduction -> melting)
    TrfFric, VZc, _ = set_day_dependent(hour, VZ, day)
    BLCond, LE, Evap = calc_blc_and_le(surf.TsurfAve, Tair, VZc, Rhz, BLCond,
                                       surf.SrfWat, DTSecs, phy)
    surf = replace(surf, EvapmmTS=Evap)
    RNet = calc_rnet(phy["Emiss"], phy["SB_const"], surf.TsurfAve, Albedo, SW, LW)
    VSH, HS, GCond = calc_hcap_hcond(TmpNw, WCont, CC, ZDpth, NLayers, DTSecs, phy, BLCond)
    condDZ, capDZ = calc_cap_cond(CC, DyK, DyC, VSH, NLayers)
    TmpNw_new, _ = calc_profile(Tmp, condDZ, capDZ, NLayers, DTSecs, TrfFric,
                                BLCond, RNet, LE)
    HStor = calc_hstor(Tmp, TmpNw_new, HS[0])
    t1, t2, _, q2 = melting(TmpNw_new[1], TmpNw_new[2], surf.TsurfAve, surf.Q2Melt,
                            surf.T4Melt, HStor, HS[0], surf.SrfSnow, surf.SrfIce,
                            surf.SrfIce2, inCouplingPhase, TsurfObsLast, cp)
    TmpNw_new[1], TmpNw_new[2] = t1, t2
    Tmp_final = TmpNw_new.copy()
    surf = replace(surf, Q2Melt=q2, TsurfAve=(Tmp_final[1] + Tmp_final[2]) / 2.0)

    # 3) traffic wear + 4) road condition (storage sequence)
    wf = wear_factors(surf.SrfSnow, surf.SrfIce, surf.SrfIce2, surf.SrfDep,
                      surf.SrfWat, Tph)
    cp2 = {**cp, "Snow2IceFac": wf.Snow2IceFac}
    rc = road_cond(surf, wf, MaxPormms, DTSecs, cp2)
    surf = rc.state_next

    # 5) albedo
    Albedo = calc_albedo(surf.WearSurf, surf.SrfSnow, surf.SrfIce, surf.SrfIce2,
                         surf.SrfDep, Albedo, cp)

    # single full-step mass audit: precipitation input + condition sequence
    step_ledger = merge_ledgers(prec_ledger, rc.ledger)

    return {"TmpNw": Tmp_final, "TsurfAve": surf.TsurfAve, "BLCond": BLCond,
            "Albedo": Albedo, "surf": surf,
            "prec_ledger": prec_ledger, "cond_ledger": rc.ledger,
            "step_ledger": step_ledger}

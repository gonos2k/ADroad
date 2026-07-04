"""Storage terms — precipitation typing & input (M2a).

Ports RoadSurf-Python `Cond.CalcPrecType` + `Storage.PrecipitationToStorage`
(Karsisto 2024, eq 42 & Table 1). Exact mode: mirrors reference branch logic.
The precipitation-type sigmoid (eq 42) is already differentiable; the hard
PLimSnow/PLimRain cut-offs are smoothed later in `smooth_compat`.

Precip phase codes (input): 0 none, 1 rain, 2 sleet, 3 snow,
4 freezing drizzle, 5 freezing rain, 6 hail.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .branches import guarded_exp
from .ledger import INTERNAL_TRANSFER_KEYS, StorageResult, make_ledger

_SNOW_DRY = 1


def _primary(wat, snow, ice, dep):
    """Primary storage mass (ice2 is auxiliary, tracked separately)."""
    return wat + snow + ice + dep


def _phase_ledger(before, after, ice2_before, ice2_after, events):
    """Ledger for a phase-change step: internal transfers conserve primary mass,
    so the net primary change (wear / clamp export) is the external flow. Kept
    consistent with water_storage's delta-as-external convention (P0 §3)."""
    delta = after - before
    id2 = ice2_after - ice2_before
    return make_ledger(
        primary_before=before,
        external_source=max(delta, 0.0),
        external_sink=max(-delta, 0.0),
        primary_after_actual=after,
        internal_transfer={k: 0.0 for k in INTERNAL_TRANSFER_KEYS},
        auxiliary_update={"ice2_increase": max(id2, 0.0),
                          "ice2_decrease": max(-id2, 0.0), "ice2_reset": 0.0},
        event_flags=events,
    )


@dataclass(frozen=True)
class Surf:
    """Surface storage state (water-equivalent mm) + fields storage logic reads."""
    SrfWat: float = 0.0
    SrfSnow: float = 0.0
    SrfIce: float = 0.0
    SrfIce2: float = 0.0
    SrfDep: float = 0.0
    TsurfAve: float = 0.0
    EvapmmTS: float = 0.0
    Q2Melt: float = 0.0
    T4Melt: float = 0.0
    WearSurf: bool = True
    SnowType: int = 1        # 1 dry, 2 wet
    WetSnowFrozen: bool = False
    SnowIceRat: float = 0.0
    VeryCold: bool = False

_NONE, _RAIN, _SLEET, _SNOW, _FDRIZZLE, _FRAIN, _HAIL = 0, 1, 2, 3, 4, 5, 6
_SNOW_WET = 2


@dataclass(frozen=True)
class PrecType:
    RainmmTS: float
    SnowmmTS: float
    PrecType: int
    SnowType: int | None      # None = leave unchanged
    PrecInTStep: float        # may be zeroed below the minimum
    RainIntensity: float
    SnowIntensity: float


def calc_prec_type(prec_in_tstep, prec_phase, Tair, Rhz, DTSecs, cp) -> PrecType:
    """cp: dict with MinPrecmm, MissValI, PLimSnow, PLimRain."""
    rain = snow = 0.0
    ptype = -1
    stype = None
    p = prec_in_tstep
    use_interp = True

    if prec_phase > cp["MissValI"]:
        use_interp = False
        if p <= cp["MinPrecmm"]:
            p, ptype, rain, snow = 0.0, -1, 0.0, 0.0
        elif prec_phase in (_NONE, _RAIN, _FDRIZZLE, _FRAIN):
            rain, snow, ptype, stype = p, 0.0, 1, _SNOW_WET
        elif prec_phase == _SLEET:
            snow = p / 2.0
            rain, ptype, stype = snow, 2, _SNOW_WET
        elif prec_phase in (_SNOW, _HAIL):
            snow, ptype, rain = p, 3, 0.0
        else:
            use_interp = True

    if use_interp:
        if p <= cp["MinPrecmm"]:
            p, ptype, rain, snow = 0.0, -1, 0.0, 0.0
        else:
            p_exp = 22.0 - 2.7 * Tair - 0.20 * Rhz
            p_rain = 1.0 / (1.0 + guarded_exp("precip.p_rain_sigmoid", p_exp))
            if p_rain < cp["PLimSnow"]:
                snow, ptype = p, 3
            elif p_rain > cp["PLimRain"]:
                rain, stype, ptype = p, _SNOW_WET, 1
            else:
                snow = p / 2.0
                rain, stype, ptype = snow, _SNOW_WET, 2

    return PrecType(
        RainmmTS=rain, SnowmmTS=snow, PrecType=ptype, SnowType=stype,
        PrecInTStep=p,
        RainIntensity=(rain / DTSecs) * 3600.0,
        SnowIntensity=(snow / DTSecs) * 3600.0,
    )


@dataclass(frozen=True)
class WearFactors:
    SnowTran: float
    IceWear: float
    IceWear2: float
    DepWear: float
    WatWear: float
    Snow2IceFac: float


def wear_factors(SrfSnow, SrfIce, SrfIce2, SrfDep, SrfWat, Tph) -> WearFactors:
    """Traffic wear factors (Karsisto 2024 §3.5.2). Mirrors Cond.WearFactors."""
    snow_tran = max((0.2 + 0.25) * SrfSnow, 0.01)
    if SrfSnow < 0.2:                    # small snow layer wears 3x faster
        snow_tran *= 3
    snow_tran *= Tph
    snow2ice = 0.25 / (0.2 + 0.25)

    ice_wear = max(1.1 * 2.0 * 0.145 * SrfIce, 0.01) * Tph
    ice_wear2 = max(1.1 * 2.0 * (4.0 * 0.290) * SrfIce2, 0.01) * Tph
    dep_wear = max(0.5 * 2.0 * (4.0 * 0.290) * SrfDep, 0.01) * Tph
    wat_wear = 10.0 * max(0.145 * SrfWat, 0.06) * Tph
    return WearFactors(snow_tran, ice_wear, ice_wear2, dep_wear, wat_wear, snow2ice)


def water_storage(SrfWat, SrfSnow, SrfIce, SrfDep, TsurfAve, EvapmmTS,
                  WearSurf, WatWear, MaxPormms, cp):
    """Water storage update: evaporation/condensation, traffic wear, limits.
    Mirrors Storage.WaterStorage. Returns (SrfWat_new, ledger).
    cp keys: TLimDew, PorEvaF, WWearLim, WWetLim, DampWearF, MinWatmms, MaxWatmms.
    """
    before = SrfWat
    w = SrfWat

    # evaporation / condensation — only on bare, warm surface
    if SrfSnow <= 0.0 and SrfIce <= 0.0 and SrfDep <= 0.0 and TsurfAve > cp["TLimDew"]:
        if w > MaxPormms:
            w -= EvapmmTS                       # surface water
        else:
            w -= cp["PorEvaF"] * EvapmmTS       # pore water

    # traffic wear (reads water level after evaporation)
    if WearSurf and w > 0.0:
        ww = 0.0 if w < cp["WWearLim"] else WatWear
        if w > cp["WWetLim"]:
            w -= ww
        else:
            w -= cp["DampWearF"] * ww

    if w < cp["MinWatmms"]:
        w = 0.0
    if w > cp["MaxWatmms"]:
        w = cp["MaxWatmms"]

    delta = w - before
    ledger = make_ledger(
        primary_before=before + SrfSnow + SrfIce + SrfDep,
        external_source=max(delta, 0.0),        # net condensation
        external_sink=max(-delta, 0.0),         # evap + wear + clamp export
        primary_after_actual=w + SrfSnow + SrfIce + SrfDep,
        internal_transfer={k: 0.0 for k in
                           ("water_to_ice", "ice_to_water", "snow_to_water",
                            "snow_to_ice", "deposit_to_water")},
        auxiliary_update={"ice2_increase": 0.0, "ice2_decrease": 0.0, "ice2_reset": 0.0},
        event_flags={"freeze_event": False, "melt_event": False,
                     "snow_event": False, "deposit_melt_event": False},
    )
    return w, ledger


def snow_storage(s: Surf, wearF, MaxPormms, DTSecs, cp) -> StorageResult:
    """Snow storage step (Storage.SnowStorage). WET=2, DRY=1.
    Returns StorageResult(state_next=Surf, ledger)."""
    WET, DRY = 2, 1
    wat, snow, ice, ice2, dep = s.SrfWat, s.SrfSnow, s.SrfIce, s.SrfIce2, s.SrfDep
    snowtype, wetfrozen = s.SnowType, s.WetSnowFrozen
    before = _primary(wat, snow, ice, dep)
    snow_in, ice2_in = snow, ice2

    ext = max(wat - MaxPormms, 0.0)
    watsnowrat = 0.0
    rd = snow + ice
    snowicerat = snow / rd if rd > 0.001 else 0.0

    if snow > 0.0:
        watsnowrat = ext / (ext + snow)
        if watsnowrat > cp["WetSnowFormR"]:
            snowtype = WET
    else:
        snowtype = DRY

    if snow > 0.0 and dep > 0.0:        # deposit under snow -> ice
        ice += dep
        dep = 0.0

    if snow > 0.0:
        if cp["forceSnowMelting"]:
            wat += snow
            snow = 0.0
        elif s.Q2Melt > 0.0 and s.TsurfAve >= cp["TLimMeltSnow"]:
            melted = (s.Q2Melt * DTSecs) / (cp["WatMHeat"] * cp["WatDens"])
            snow -= 1000.0 * melted
            wat += 1000.0 * melted

    if s.WearSurf and snow > 0.0:       # wear: snow -> ice
        snow -= wearF.SnowTran
        ice += cp["Snow2IceFac"] * wearF.SnowTran
        ice2 += cp["Snow2IceFac"] * wearF.SnowTran

    if snow > 0.0 and snowtype == WET:
        if watsnowrat > cp["WetSnowMeltR"]:
            wat += snow
            snow = 0.0
            snowtype = DRY
        if s.TsurfAve < cp["TLimFreeze"]:
            ice += snow + wat
            ice2 += snow + wat
            snowtype = DRY
            if snow > 0.5:
                wetfrozen = True
            snow = 0.0
            wat = 0.0

    if snow < cp["MinSnowmms"]:
        snow = 0.0
    if snow > cp["MaxSnowmms"]:
        snow -= cp["MaxSnowmms"] / 2.0

    s_next = replace(s, SrfWat=wat, SrfSnow=snow, SrfIce=ice, SrfIce2=ice2, SrfDep=dep,
                     SnowType=snowtype, WetSnowFrozen=wetfrozen, SnowIceRat=snowicerat)
    ledger = _phase_ledger(before, _primary(wat, snow, ice, dep), ice2_in, ice2,
                           {"freeze_event": wetfrozen or snow_in > snow,
                            "melt_event": wat > s.SrfWat, "snow_event": False,
                            "deposit_melt_event": False})
    return StorageResult(s_next, ledger)


def ice_storage(s: Surf, wearF, DTSecs, cp) -> StorageResult:
    """Ice storage step (Storage.IceStorage). Returns StorageResult."""
    wat, snow, ice, ice2 = s.SrfWat, s.SrfSnow, s.SrfIce, s.SrfIce2
    before = _primary(wat, snow, ice, s.SrfDep)
    ice2_in = ice2

    if s.TsurfAve < cp["TLimFreeze"] and wat > 0.0:     # freezing
        ice += wat
        ice2 += wat
        wat = 0.0

    if snow <= 0.0 and ice > 0.0:                        # melting (snow-free)
        if cp["forceIceMelting"]:
            wat += ice
            ice = 0.0
            ice2 = 0.0
        elif s.Q2Melt > 0.0 and s.TsurfAve >= cp["TLimMeltIce"]:
            melted = (s.Q2Melt * DTSecs) / (cp["WatMHeat"] * cp["WatDens"])
            ice -= 1000.0 * melted
            ice2 -= 1000.0 * melted
            wat += 1000.0 * melted

    if s.WearSurf and ice > 0.0:
        ice -= wearF.IceWear
    if s.WearSurf and ice2 > 0.0:
        ice2 -= wearF.IceWear2

    if ice < cp["MinIcemms"]:
        ice = 0.0
    if ice > cp["MaxIcemms"]:
        ice = cp["MaxIcemms"]
    if ice2 < cp["MinIcemms"]:
        ice2 = 0.0
    if ice2 > cp["MaxIcemms"]:
        ice2 = cp["MaxIcemms"]

    s_next = replace(s, SrfWat=wat, SrfIce=ice, SrfIce2=ice2)
    ledger = _phase_ledger(before, _primary(wat, snow, ice, s.SrfDep), ice2_in, ice2,
                           {"freeze_event": ice > s.SrfIce, "melt_event": wat > s.SrfWat,
                            "snow_event": False, "deposit_melt_event": False})
    return StorageResult(s_next, ledger)


def deposit_storage(s: Surf, DepWear, cp) -> StorageResult:
    """Deposit (black ice) storage step (Storage.DepositStorage). Returns StorageResult."""
    wat, snow, dep = s.SrfWat, s.SrfSnow, s.SrfDep
    before = _primary(wat, snow, s.SrfIce, dep)
    dep_in = dep

    if s.EvapmmTS < 0.0:                 # condensation only
        dep -= s.EvapmmTS
    if s.TsurfAve > cp["TLimMeltDep"]:   # melting -> water
        wat += dep
        dep = 0.0
    if s.WearSurf and snow <= 0.0 and dep > 0.0:
        dep -= DepWear
    if dep < cp["MinDepmms"]:
        dep = 0.0
    if dep > cp["MaxDepmms"]:            # overflow -> water
        wat += dep - cp["MaxDepmms"]
        dep = cp["MaxDepmms"]

    s_next = replace(s, SrfWat=wat, SrfDep=dep)
    ledger = _phase_ledger(before, _primary(wat, snow, s.SrfIce, dep), s.SrfIce2, s.SrfIce2,
                           {"freeze_event": False, "melt_event": False, "snow_event": False,
                            "deposit_melt_event": dep < dep_in})
    return StorageResult(s_next, ledger)


def new_melt_freeze_heat(s: Surf, DTSecs, cp) -> Surf:
    """Heat needed to melt uppermost snow/ice layer (Storage.NewMeltFreezeHeat)."""
    q = 0.0
    t4 = s.T4Melt
    if s.SrfSnow > 0.0:
        q = cp["WatMHeat"] * cp["WatDens"] * (s.SrfSnow / 1000.0) / DTSecs
        t4 = cp["TLimMeltSnow"]
    if s.SrfSnow <= 0.0 and s.SrfIce > 0.0:
        q = cp["WatMHeat"] * cp["WatDens"] * (s.SrfIce / 1000.0) / DTSecs
        t4 = cp["TLimMeltIce"]
    if q < 0.0:
        q = 0.0
    return replace(s, Q2Melt=q, T4Melt=t4)


def melting(TmpNw1, TmpNw2, TsurfAve, Q2Melt, T4Melt, HStor, HS0,
            snow, ice, ice2, inCouplingPhase, TsurfObsLast, cp):
    """Melting heat balance coupled to ground temp (BalanceModel.melting).
    Returns (TmpNw1, TmpNw2, TsurfAve, Q2Melt). With CanMeltingChangeTemperature
    False (default) ground temperature is not modified."""
    outer = ((snow > 0.0 or ice > 0.0 or ice2 > 0.0) and
             (HStor > 0.00001 and TsurfAve > T4Melt and Q2Melt > 0
              and (not inCouplingPhase or TsurfObsLast < T4Melt)))
    if not outer:
        return TmpNw1, TmpNw2, TsurfAve, 0.0

    if not cp["CanMeltingChangeTemperature"]:
        return TmpNw1, TmpNw2, TsurfAve, Q2Melt

    # redundant inner guard (kept for exact fidelity)
    if (HStor <= 0.00001 or TsurfAve <= T4Melt or Q2Melt <= 0
            or (inCouplingPhase and TsurfObsLast < T4Melt)):
        if TsurfAve < 0.5:
            return TmpNw1, TmpNw2, TsurfAve, 0.0
        if TsurfAve > 2.0:
            QAvail = HS0 * (TmpNw1 - T4Melt)
            return TmpNw1, TmpNw2, TsurfAve, QAvail

    QAvail = HS0 * (TmpNw1 - T4Melt)
    if Q2Melt >= QAvail:
        Q2Melt = QAvail
        TmpNw1 = T4Melt + 0.01
        TmpNw2 = T4Melt + 0.01
    else:
        QLeftOver = QAvail - Q2Melt
        TmpNw1 = T4Melt + QLeftOver / HS0
        TmpNw2 = T4Melt + 0.01
    TsurfAve = 0.5 * (TmpNw1 + TmpNw2)
    return TmpNw1, TmpNw2, TsurfAve, Q2Melt


def calc_albedo(WearSurf, snow, ice, ice2, dep, current, cp):
    """Surface albedo (Cond.CalcAlbedo). If WearSurf is False, keep current."""
    if not WearSurf:
        return current
    icesum = 0.5 * (ice + ice2) + dep
    if icesum < 0.0:
        icesum = 0.0
    icemax = 1.5
    alb = cp["AlbDry"]
    if snow > 0.01 and snow > ice:
        alb = cp["AlbSnow"]
    elif ice > 0.01 or dep > 0.01:
        if icesum < icemax:
            alb = cp["AlbDry"] + (icesum / icemax) * (cp["AlbSnow"] - cp["AlbDry"])
        else:
            alb = cp["AlbSnow"]
    return alb


def precipitation_to_storage(SrfWat, SrfSnow, SrfIce, SrfDep, pt: PrecType):
    """Add rain to water storage, snow to snow storage. Returns (SrfWat, SrfSnow, ledger)."""
    primary_before = SrfWat + SrfSnow + SrfIce + SrfDep
    SrfWat_new = SrfWat + pt.RainmmTS
    SrfSnow_new = SrfSnow + pt.SnowmmTS
    primary_after = SrfWat_new + SrfSnow_new + SrfIce + SrfDep

    ledger = make_ledger(
        primary_before=primary_before,
        external_source=pt.RainmmTS + pt.SnowmmTS,
        external_sink=0.0,
        primary_after_actual=primary_after,
        internal_transfer={k: 0.0 for k in
                           ("water_to_ice", "ice_to_water", "snow_to_water",
                            "snow_to_ice", "deposit_to_water")},
        auxiliary_update={"ice2_increase": 0.0, "ice2_decrease": 0.0, "ice2_reset": 0.0},
        event_flags={"freeze_event": False, "melt_event": False,
                     "snow_event": pt.SnowmmTS > 0.0, "deposit_melt_event": False},
    )
    return SrfWat_new, SrfSnow_new, ledger

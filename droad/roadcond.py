"""RoadCond assembly (Cond.RoadCond): sequence the storage updates.

Order mirrors reference: VeryCold hysteresis -> water -> snow -> ice -> deposit
-> water re-clamp -> melt/freeze heat. SnowType reset to dry at entry.

Each sub-step returns a ledger; road_cond aggregates them with merge_ledgers so
the whole condition step is a single, mass-auditable StorageResult (P0 §3).
"""

from __future__ import annotations

from dataclasses import replace

from .ledger import (
    DIAG_WATER_NEGATIVE_PRE_CLAMP, DIAG_WATER_OVERFLOW,
    INTERNAL_TRANSFER_KEYS, StorageResult, make_ledger, merge_ledgers,
)
from .storage import (
    water_storage, snow_storage, ice_storage, deposit_storage, new_melt_freeze_heat,
)


def _reclamp_ledger(before, after):
    """Ledger for the water re-clamp (export/import via Min/Max limits)."""
    delta = after - before
    return make_ledger(
        primary_before=before, external_source=max(delta, 0.0),
        external_sink=max(-delta, 0.0), primary_after_actual=after,
        internal_transfer={k: 0.0 for k in INTERNAL_TRANSFER_KEYS},
        auxiliary_update={"ice2_increase": 0.0, "ice2_decrease": 0.0, "ice2_reset": 0.0},
        event_flags={"freeze_event": False, "melt_event": False,
                     "snow_event": False, "deposit_melt_event": False},
    )


def _prim(s):
    return s.SrfWat + s.SrfSnow + s.SrfIce + s.SrfDep


def road_cond(s, wf, MaxPormms, DTSecs, cp) -> StorageResult:
    """Returns StorageResult(state_next=Surf, ledger) — ledger aggregates the
    water/snow/ice/deposit/re-clamp sub-steps via merge_ledgers."""
    s = replace(s, SnowType=1, SnowIceRat=0.0)     # reset (atm.SnowType=DRY)

    vc = s.VeryCold                                  # hysteresis (TLimColdH > TLimColdL)
    if vc and s.TsurfAve > cp["TLimColdH"]:
        vc = False
    if (not vc) and s.TsurfAve < cp["TLimColdL"]:
        vc = True
    s = replace(s, VeryCold=vc)

    w, lw, wdiag = water_storage(s.SrfWat, s.SrfSnow, s.SrfIce, s.SrfDep, s.TsurfAve,
                                 s.EvapmmTS, s.WearSurf, wf.WatWear, MaxPormms, cp)
    s = replace(s, SrfWat=w)

    rs = snow_storage(s, wf, MaxPormms, DTSecs, cp); s = rs.state_next
    ri = ice_storage(s, wf, DTSecs, cp); s = ri.state_next
    rd = deposit_storage(s, wf.DepWear, cp); s = rd.state_next
    diagnostics = wdiag + rs.diagnostics + ri.diagnostics + rd.diagnostics

    before_clamp = _prim(s)                          # water limits re-check
    w = s.SrfWat
    if w < 0.0:
        diagnostics = diagnostics + (DIAG_WATER_NEGATIVE_PRE_CLAMP,)
    if w > cp["MaxWatmms"]:
        diagnostics = diagnostics + (DIAG_WATER_OVERFLOW,)
    if w < cp["MinWatmms"]:
        w = 0.0
    if w > cp["MaxWatmms"]:
        w = cp["MaxWatmms"]
    s = replace(s, SrfWat=w)
    lclamp = _reclamp_ledger(before_clamp, _prim(s))

    s = new_melt_freeze_heat(s, DTSecs, cp)          # no primary-mass change
    ledger = merge_ledgers(lw, rs.ledger, ri.ledger, rd.ledger, lclamp)
    return StorageResult(s, ledger, diagnostics)

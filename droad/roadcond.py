"""RoadCond assembly (Cond.RoadCond): sequence the storage updates.

Order mirrors reference: VeryCold hysteresis -> water -> snow -> ice -> deposit
-> water re-clamp -> melt/freeze heat. SnowType reset to dry at entry.
"""

from __future__ import annotations

from dataclasses import replace

from .storage import (
    water_storage, snow_storage, ice_storage, deposit_storage, new_melt_freeze_heat,
)


def road_cond(s, wf, MaxPormms, DTSecs, cp):
    """Returns updated Surf."""
    s = replace(s, SnowType=1, SnowIceRat=0.0)     # reset (atm.SnowType=DRY)

    vc = s.VeryCold                                  # hysteresis (TLimColdH > TLimColdL)
    if vc and s.TsurfAve > cp["TLimColdH"]:
        vc = False
    if (not vc) and s.TsurfAve < cp["TLimColdL"]:
        vc = True
    s = replace(s, VeryCold=vc)

    w, _ = water_storage(s.SrfWat, s.SrfSnow, s.SrfIce, s.SrfDep, s.TsurfAve,
                         s.EvapmmTS, s.WearSurf, wf.WatWear, MaxPormms, cp)
    s = replace(s, SrfWat=w)

    s = snow_storage(s, wf, MaxPormms, DTSecs, cp)
    s = ice_storage(s, wf, DTSecs, cp)
    s = deposit_storage(s, wf.DepWear, cp)

    w = s.SrfWat                                      # water limits re-check
    if w < cp["MinWatmms"]:
        w = 0.0
    if w > cp["MaxWatmms"]:
        w = cp["MaxWatmms"]
    s = replace(s, SrfWat=w)

    return new_melt_freeze_heat(s, DTSecs, cp)

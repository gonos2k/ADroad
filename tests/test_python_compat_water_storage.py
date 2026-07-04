"""G2 python_compat (M2b): wear_factors & water_storage vs RoadSurf-Python.

Decoupled unit tests over regime-spanning synthetic surface states (dry/damp/wet,
warm/cold, snow present/absent, wear on/off, evaporation vs condensation).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.storage import wear_factors, water_storage  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    return m, objs[8]  # modules, condParam


def _cp_dict(cp):
    return {"TLimDew": cp.TLimDew, "PorEvaF": cp.PorEvaF, "WWearLim": cp.WWearLim,
            "WWetLim": cp.WWetLim, "DampWearF": cp.DampWearF,
            "MinWatmms": cp.MinWatmms, "MaxWatmms": cp.MaxWatmms}


WEAR_CASES = [
    # SrfSnow, SrfIce, SrfIce2, SrfDep, SrfWat
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.1, 0.05, 0.02, 0.01, 0.5),     # small snow (<0.2 -> x3)
    (5.0, 2.0, 1.0, 1.5, 3.0),        # large storages
    (0.2, 0.0, 0.0, 0.0, 0.0),        # snow exactly at 0.2 boundary
]


@pytest.mark.parametrize("st", WEAR_CASES)
def test_wear_factors_matches_reference(ref, st):
    m, cp = ref
    SrfSnow, SrfIce, SrfIce2, SrfDep, SrfWat = st
    Tph = 30.0 / 3600.0
    surf = SimpleNamespace(SrfSnowmms=SrfSnow, SrfIcemms=SrfIce, SrfIce2mms=SrfIce2,
                           SrfDepmms=SrfDep, SrfWatmms=SrfWat)
    wf_ref = m["WearingFactors"].WearingFactors()
    cp_ref = SimpleNamespace(Snow2IceFac=0.0)
    m["Cond"].WearFactors(cp_ref, Tph, surf, wf_ref)

    wf = wear_factors(SrfSnow, SrfIce, SrfIce2, SrfDep, SrfWat, Tph)
    assert wf.SnowTran == pytest.approx(wf_ref.SnowTran, abs=1e-12)
    assert wf.IceWear == pytest.approx(wf_ref.IceWear, abs=1e-12)
    assert wf.IceWear2 == pytest.approx(wf_ref.IceWear2, abs=1e-12)
    assert wf.DepWear == pytest.approx(wf_ref.DepWear, abs=1e-12)
    assert wf.WatWear == pytest.approx(wf_ref.WatWear, abs=1e-12)
    assert wf.Snow2IceFac == pytest.approx(cp_ref.Snow2IceFac, abs=1e-12)


WATER_CASES = [
    # SrfWat, SrfSnow, SrfIce, SrfDep, TsurfAve, EvapmmTS, WearSurf, WatWear
    (0.0, 0.0, 0.0, 0.0, 5.0, 0.01, True, 0.05),      # dry, warm, evap
    (0.5, 0.0, 0.0, 0.0, 5.0, 0.02, True, 0.03),      # pore water, evap, wear
    (1.5, 0.0, 0.0, 0.0, 5.0, 0.05, True, 0.1),       # surface water (>MaxPor)
    (0.95, 0.0, 0.0, 0.0, 5.0, 0.0, True, 0.2),       # wet (>WWetLim) full wear
    (0.5, 0.0, 0.0, 0.0, 5.0, 0.0, True, 0.2),        # damp (<WWetLim) half wear
    (0.05, 0.0, 0.0, 0.0, 5.0, 0.0, True, 0.2),       # below WWearLim -> no wear
    (0.5, 1.0, 0.0, 0.0, 5.0, 0.1, True, 0.05),       # snow present -> no evap
    (0.5, 0.0, 0.0, 0.0, -2.0, 0.1, True, 0.05),      # cold -> no evap
    (0.5, 0.0, 0.0, 0.0, 5.0, -0.1, True, 0.05),      # condensation (evap<0)
    (3.0, 0.0, 0.0, 0.0, 5.0, 0.0, False, 0.0),       # overflow -> MaxWatmms
    (0.5, 0.0, 0.0, 0.0, 5.0, 0.05, False, 0.2),      # WearSurf False
]


@pytest.mark.parametrize("c", WATER_CASES)
def test_water_storage_matches_reference(ref, c):
    m, cp = ref
    SrfWat, SrfSnow, SrfIce, SrfDep, Tsurf, Evap, wear, watwear = c
    MaxPor = 1.0

    surf = SimpleNamespace(SrfWatmms=SrfWat, SrfSnowmms=SrfSnow, SrfIcemms=SrfIce,
                           SrfDepmms=SrfDep, TsurfAve=Tsurf, EvapmmTS=Evap,
                           WearSurf=wear)
    m["Storage"].WaterStorage(MaxPor, watwear, surf, cp)   # mutates surf.SrfWatmms

    w, lg = water_storage(SrfWat, SrfSnow, SrfIce, SrfDep, Tsurf, Evap,
                          wear, watwear, MaxPor, _cp_dict(cp))
    assert w == pytest.approx(surf.SrfWatmms, abs=1e-12), f"case {c}"
    assert lg.primary_mass_residual == pytest.approx(0.0, abs=1e-12)

"""G2 python_compat (M2c): snow/ice/deposit storage & melt-heat vs RoadSurf-Python.

Branch-heavy phase-change logic. Each droad function is compared to its reference
counterpart on synthetic surface states chosen to hit each branch.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.storage import (  # noqa: E402
    Surf, snow_storage, ice_storage, deposit_storage, new_melt_freeze_heat,
)

DT = 30.0
SNOW2ICE = 0.25 / (0.2 + 0.25)


@pytest.fixture(scope="module")
def env():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    return m, objs[8]  # modules, condParam


def _cp(cp):
    return {
        "WetSnowFormR": cp.WetSnowFormR, "WetSnowMeltR": cp.WetSnowMeltR,
        "TLimFreeze": cp.TLimFreeze, "TLimMeltSnow": cp.TLimMeltSnow,
        "TLimMeltIce": cp.TLimMeltIce, "TLimMeltDep": cp.TLimMeltDep,
        "MinSnowmms": cp.MinSnowmms, "MaxSnowmms": cp.MaxSnowmms,
        "MinIcemms": cp.MinIcemms, "MaxIcemms": cp.MaxIcemms,
        "MinDepmms": cp.MinDepmms, "MaxDepmms": cp.MaxDepmms,
        "WatMHeat": cp.WatMHeat, "WatDens": cp.WatDens,
        "Snow2IceFac": SNOW2ICE, "forceSnowMelting": False, "forceIceMelting": False,
    }


def _ref_surf(s: Surf):
    return SimpleNamespace(
        SrfWatmms=s.SrfWat, SrfSnowmms=s.SrfSnow, SrfIcemms=s.SrfIce,
        SrfIce2mms=s.SrfIce2, SrfDepmms=s.SrfDep, TsurfAve=s.TsurfAve,
        EvapmmTS=s.EvapmmTS, Q2Melt=s.Q2Melt, T4Melt=s.T4Melt, WearSurf=s.WearSurf)


def _wearF(SnowTran=0.05, IceWear=0.02, IceWear2=0.08, WatWear=0.06):
    return SimpleNamespace(SnowTran=SnowTran, IceWear=IceWear, IceWear2=IceWear2,
                           DepWear=0.04, WatWear=WatWear)


def _assert_storages(got: Surf, rsurf):
    assert got.SrfWat == pytest.approx(rsurf.SrfWatmms, abs=1e-12)
    assert got.SrfSnow == pytest.approx(rsurf.SrfSnowmms, abs=1e-12)
    assert got.SrfIce == pytest.approx(rsurf.SrfIcemms, abs=1e-12)
    assert got.SrfIce2 == pytest.approx(rsurf.SrfIce2mms, abs=1e-12)
    assert got.SrfDep == pytest.approx(rsurf.SrfDepmms, abs=1e-12)


SNOW_CASES = [
    Surf(SrfSnow=2.0, SrfWat=0.0, TsurfAve=1.0, Q2Melt=5e3, WearSurf=True),   # melt
    Surf(SrfSnow=1.0, SrfWat=3.0, TsurfAve=1.0, Q2Melt=0.0, WearSurf=True),   # wet -> water
    Surf(SrfSnow=1.0, SrfWat=3.0, TsurfAve=-2.0, Q2Melt=0.0, WearSurf=True),  # wet + freeze
    Surf(SrfSnow=1.0, SrfDep=0.5, TsurfAve=-1.0, WearSurf=True),              # dep under snow
    Surf(SrfSnow=200.0, TsurfAve=-5.0, WearSurf=False),                       # overflow
    Surf(SrfSnow=0.0, SrfIce=1.0, TsurfAve=-1.0),                             # no snow
]


@pytest.mark.parametrize("s", SNOW_CASES)
def test_snow_storage(env, s):
    m, cp = env
    rs, atm = _ref_surf(s), SimpleNamespace(SnowType=s.SnowType)
    cp.Snow2IceFac, cp.forceSnowMelting = SNOW2ICE, False
    m["Storage"].SnowStorage(0.0, DT, _wearF(), 1.0, rs, cp, atm)
    got = snow_storage(s, _wearF(), 1.0, DT, _cp(cp))
    _assert_storages(got, rs)
    assert got.SnowType == atm.SnowType


ICE_CASES = [
    Surf(SrfWat=2.0, TsurfAve=-2.0),                                   # freezing
    Surf(SrfIce=2.0, SrfIce2=2.0, SrfSnow=0.0, TsurfAve=1.0, Q2Melt=5e3),  # melting
    Surf(SrfIce=1.0, SrfIce2=1.0, SrfSnow=1.0, TsurfAve=1.0, WearSurf=True),  # snow present, wear only
    Surf(SrfIce=100.0, SrfIce2=100.0, TsurfAve=-5.0, WearSurf=False),  # overflow clamp
]


@pytest.mark.parametrize("s", ICE_CASES)
def test_ice_storage(env, s):
    m, cp = env
    rs = _ref_surf(s)
    cp.forceIceMelting = False
    m["Storage"].IceStorage(0.0, DT, rs, cp, _wearF())
    got = ice_storage(s, _wearF(), DT, _cp(cp))
    _assert_storages(got, rs)


DEP_CASES = [
    Surf(SrfDep=0.5, EvapmmTS=-0.1, TsurfAve=-1.0, WearSurf=True),   # condensation
    Surf(SrfDep=0.5, TsurfAve=2.0, WearSurf=True),                    # melt -> water
    Surf(SrfDep=0.5, SrfSnow=0.0, TsurfAve=-1.0, WearSurf=True),      # wear
    Surf(SrfDep=5.0, TsurfAve=-1.0, WearSurf=False),                  # overflow -> water
]


@pytest.mark.parametrize("s", DEP_CASES)
def test_deposit_storage(env, s):
    m, cp = env
    rs = _ref_surf(s)
    m["Storage"].DepositStorage(_wearF().DepWear, rs, cp)
    got = deposit_storage(s, _wearF().DepWear, _cp(cp))
    _assert_storages(got, rs)


MELT_CASES = [
    Surf(SrfSnow=1.0), Surf(SrfSnow=0.0, SrfIce=2.0), Surf(SrfSnow=0.0, SrfIce=0.0),
]


@pytest.mark.parametrize("s", MELT_CASES)
def test_new_melt_freeze_heat(env, s):
    m, cp = env
    rs = _ref_surf(s)
    m["Storage"].NewMeltFreezeHeat(DT, rs, cp)
    got = new_melt_freeze_heat(s, DT, _cp(cp))
    assert got.Q2Melt == pytest.approx(rs.Q2Melt, abs=1e-9)
    assert got.T4Melt == pytest.approx(rs.T4Melt, abs=1e-12)

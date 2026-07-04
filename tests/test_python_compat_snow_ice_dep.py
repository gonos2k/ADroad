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
    got = snow_storage(s, _wearF(), 1.0, DT, _cp(cp)).state_next
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
    got = ice_storage(s, _wearF(), DT, _cp(cp)).state_next
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
    got = deposit_storage(s, _wearF().DepWear, _cp(cp)).state_next
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


def _cp_synthetic():
    return {"WetSnowFormR": 0.3, "WetSnowMeltR": 0.6, "TLimFreeze": -0.5,
            "TLimMeltSnow": 0.0, "TLimMeltIce": 0.0, "TLimMeltDep": 0.0,
            "MinSnowmms": 0.001, "MaxSnowmms": 200.0, "MinIcemms": 0.001,
            "MaxIcemms": 100.0, "MinDepmms": 0.001, "MaxDepmms": 2.0,
            "WatMHeat": 3.34e5, "WatDens": 1000.0, "Snow2IceFac": SNOW2ICE,
            "forceSnowMelting": False, "forceIceMelting": False}


def test_storage_functions_return_ledger():
    """R2: snow/ice/deposit return a well-formed StorageResult ledger."""
    from droad.ledger import StorageResult, StorageLedger
    from droad.storage import Surf, snow_storage, ice_storage, deposit_storage
    cp = _cp_synthetic()
    s = Surf(SrfWat=1.0, SrfSnow=2.0, SrfIce=0.5, SrfDep=0.3, TsurfAve=-1.0, WearSurf=True)
    for r in (snow_storage(s, _wearF(), 1.0, DT, cp),
              ice_storage(s, _wearF(), DT, cp),
              deposit_storage(s, _wearF().DepWear, cp)):
        assert isinstance(r, StorageResult)
        assert isinstance(r.state_next, Surf)
        assert isinstance(r.ledger, StorageLedger)              # keys validated in __post_init__


def test_ice_freeze_transfer_amount_and_flag():
    """R2 teeth: freezing books water_to_ice == frozen water and sets freeze_event."""
    from droad.storage import Surf, ice_storage
    cp = _cp_synthetic()
    r = ice_storage(Surf(SrfWat=2.0, TsurfAve=-2.0, WearSurf=False), _wearF(), DT, cp)
    assert r.ledger.internal_transfer["water_to_ice"] == pytest.approx(2.0)
    assert r.ledger.internal_transfer["ice_to_water"] == 0.0
    assert r.ledger.event_flags["freeze_event"] is True
    assert r.ledger.event_flags["melt_event"] is False
    assert r.ledger.auxiliary_update["ice2_increase"] == pytest.approx(2.0)  # ice2 += wat


def test_deposit_melt_transfer_amount_and_flag():
    """R2 teeth: deposit melting books deposit_to_water and sets deposit_melt_event."""
    from droad.storage import Surf, deposit_storage
    cp = _cp_synthetic()
    r = deposit_storage(Surf(SrfDep=0.5, TsurfAve=2.0, WearSurf=False), _wearF().DepWear, cp)
    assert r.ledger.internal_transfer["deposit_to_water"] == pytest.approx(0.5)
    assert r.ledger.event_flags["deposit_melt_event"] is True


def test_snow_wear_books_snow_to_ice():
    """R2 teeth: traffic wear on a cold dry snowpack books the conserved snow_to_ice
    part (= Snow2IceFac*SnowTran), and does NOT falsely flag a melt/freeze event."""
    from droad.storage import Surf, snow_storage
    cp = _cp_synthetic()
    wf = _wearF(SnowTran=0.05)
    r = snow_storage(Surf(SrfSnow=5.0, TsurfAve=-3.0, WearSurf=True), wf, 1.0, DT, cp)
    assert r.ledger.internal_transfer["snow_to_ice"] == pytest.approx(SNOW2ICE * 0.05)
    assert r.ledger.event_flags["melt_event"] is False
    assert r.ledger.event_flags["freeze_event"] is False


# --- adversarial regression (3rd review): residual is a real leak detector ---

def _all_residuals_zero(*results):
    return all(abs(r.ledger.primary_mass_residual) < 1e-9 for r in results)


def test_external_accounting_makes_residual_zero_for_correct_code():
    """Correct storage steps (wear/clamp/condensation booked as external) leave a
    ~0 residual — so a NON-zero residual would signal an unaccounted leak."""
    from droad.storage import Surf, snow_storage, ice_storage, deposit_storage
    cp = _cp_synthetic()
    cases = [
        Surf(SrfSnow=5.0, TsurfAve=-3.0, WearSurf=True),          # snow wear + clamp
        Surf(SrfSnow=300.0, TsurfAve=-5.0, WearSurf=False),       # snow overflow export
        Surf(SrfWat=2.0, TsurfAve=-2.0, WearSurf=True),           # ice freeze
        Surf(SrfIce=1.0, SrfIce2=1.0, TsurfAve=1.0, Q2Melt=5e3),  # ice melt + wear
        Surf(SrfDep=0.5, EvapmmTS=-0.1, TsurfAve=-1.0, WearSurf=True),  # deposit condensation+wear
        Surf(SrfDep=5.0, TsurfAve=-1.0, WearSurf=False),          # deposit overflow -> water
    ]
    for s in cases:
        assert _all_residuals_zero(
            snow_storage(s, _wearF(), 1.0, DT, cp),
            ice_storage(s, _wearF(), DT, cp),
            deposit_storage(s, _wearF().DepWear, cp))


def test_phase_ledger_rejects_transfer_typo():
    """A mistyped transfer key must raise, not be silently dropped."""
    from droad.ledger import LedgerError
    from droad.storage import _phase_ledger
    with pytest.raises(LedgerError):
        _phase_ledger(0.0, 0.0, 0.0, 0.0, {"snow_to_icee": 1.0}, 0.0, 0.0, 0.0,
                      {"freeze_event": False, "melt_event": False,
                       "snow_event": False, "deposit_melt_event": False})


def test_phase_ledger_surfaces_unaccounted_leak():
    """If a step changes mass without booking it as transfer/external, the residual
    is non-zero (the whole point of branch-local accounting)."""
    from droad.storage import _phase_ledger
    # before=2, after=1.5, but nothing booked as external sink -> 0.5 unexplained
    lg = _phase_ledger(2.0, 1.5, 0.0, 0.0, {}, 0.0, 0.0, 0.0,
                       {"freeze_event": False, "melt_event": False,
                        "snow_event": False, "deposit_melt_event": False})
    assert lg.primary_mass_residual == pytest.approx(-0.5)


def test_ice2_reset_recorded_on_force_melt():
    """forceIceMelting zeroes ice2 -> recorded as ice2_reset, not just decrease."""
    from droad.storage import Surf, ice_storage
    cp = {**_cp_synthetic(), "forceIceMelting": True}
    r = ice_storage(Surf(SrfIce=1.0, SrfIce2=0.8, SrfSnow=0.0, TsurfAve=1.0), _wearF(), DT, cp)
    assert r.ledger.auxiliary_update["ice2_reset"] == pytest.approx(0.8)


# --- feasibility diagnostics (separate from mass residual) ---

def test_over_melt_diagnostics_but_residual_zero():
    """Energy-unlimited melt (amt > available) is flagged in diagnostics, yet the
    mass ledger residual stays ~0 (the clamp import is booked as external)."""
    from droad.storage import Surf, snow_storage, ice_storage
    cp = _cp_synthetic()
    rs = snow_storage(Surf(SrfSnow=0.5, TsurfAve=1.0, Q2Melt=1e5, WearSurf=False),
                      _wearF(), 1.0, DT, cp)
    assert "snow_over_melt" in rs.diagnostics
    assert "snow_negative_pre_clamp" in rs.diagnostics
    assert abs(rs.ledger.primary_mass_residual) < 1e-9

    ri = ice_storage(Surf(SrfIce=0.5, SrfIce2=0.5, SrfSnow=0.0, TsurfAve=1.0, Q2Melt=1e5),
                     _wearF(), DT, cp)
    assert "ice_over_melt" in ri.diagnostics
    assert abs(ri.ledger.primary_mass_residual) < 1e-9


def test_overflow_diagnostics():
    from droad.storage import Surf, snow_storage, deposit_storage
    cp = _cp_synthetic()
    rs = snow_storage(Surf(SrfSnow=300.0, TsurfAve=-5.0, WearSurf=False), _wearF(), 1.0, DT, cp)
    assert "snow_overflow" in rs.diagnostics
    rd = deposit_storage(Surf(SrfDep=5.0, TsurfAve=-1.0, WearSurf=False), _wearF().DepWear, cp)
    assert "deposit_overflow" in rd.diagnostics


def test_road_cond_surfaces_diagnostics():
    """road_cond aggregates child diagnostics into its StorageResult."""
    from droad.storage import Surf, wear_factors
    from droad.roadcond import road_cond
    cp = {"WetSnowFormR": 0.3, "WetSnowMeltR": 0.6, "TLimFreeze": -0.5,
          "TLimMeltSnow": 0.0, "TLimMeltIce": 0.0, "TLimMeltDep": 0.0,
          "TLimColdH": -2.0, "TLimColdL": -4.0, "TLimDew": 0.0, "PorEvaF": 0.5,
          "WWearLim": 0.05, "WWetLim": 0.5, "DampWearF": 0.5, "MinWatmms": 0.001,
          "MaxWatmms": 1.0, "MinSnowmms": 0.001, "MaxSnowmms": 200.0,
          "MinIcemms": 0.001, "MaxIcemms": 100.0, "MinDepmms": 0.001,
          "MaxDepmms": 2.0, "WatMHeat": 3.34e5, "WatDens": 1000.0,
          "Snow2IceFac": SNOW2ICE, "forceSnowMelting": False, "forceIceMelting": False}
    s = Surf(SrfSnow=0.5, TsurfAve=1.0, Q2Melt=1e5, WearSurf=False)
    r = road_cond(s, wear_factors(s.SrfSnow, s.SrfIce, s.SrfIce2, s.SrfDep, s.SrfWat, 1.0),
                  1.0, DT, cp)
    assert "snow_over_melt" in r.diagnostics


def test_road_cond_aggregates_ledger():
    """R2: road_cond returns a StorageResult whose ledger merges the sub-steps."""
    from droad.ledger import StorageResult
    from droad.roadcond import road_cond
    from droad.storage import Surf, wear_factors
    cp = {"WetSnowFormR": 0.3, "WetSnowMeltR": 0.6, "TLimFreeze": -0.5,
          "TLimMeltSnow": 0.0, "TLimMeltIce": 0.0, "TLimMeltDep": 0.0,
          "TLimColdH": -2.0, "TLimColdL": -4.0, "TLimDew": 0.0, "PorEvaF": 0.5,
          "WWearLim": 0.05, "WWetLim": 0.5, "DampWearF": 0.5, "MinWatmms": 0.001,
          "MaxWatmms": 1.0, "MinSnowmms": 0.001, "MaxSnowmms": 200.0,
          "MinIcemms": 0.001, "MaxIcemms": 100.0, "MinDepmms": 0.001,
          "MaxDepmms": 2.0, "WatMHeat": 3.34e5, "WatDens": 1000.0,
          "Snow2IceFac": SNOW2ICE, "forceSnowMelting": False, "forceIceMelting": False}
    s = Surf(SrfWat=0.5, SrfSnow=1.0, SrfIce=0.2, TsurfAve=-1.0, WearSurf=True)
    r = road_cond(s, wear_factors(s.SrfSnow, s.SrfIce, s.SrfIce2, s.SrfDep, s.SrfWat, 1.0),
                  1.0, DT, cp)
    assert isinstance(r, StorageResult)
    # merged span: before = entry primary mass, after = final primary mass
    entry = s.SrfWat + s.SrfSnow + s.SrfIce + s.SrfDep
    final = r.state_next.SrfWat + r.state_next.SrfSnow + r.state_next.SrfIce + r.state_next.SrfDep
    assert r.ledger.primary_before == pytest.approx(entry)
    assert r.ledger.primary_after_actual == pytest.approx(final)
    # residual telescopes to ~0 (external flows account for the net change)
    assert abs(r.ledger.primary_mass_residual) < 1e-9

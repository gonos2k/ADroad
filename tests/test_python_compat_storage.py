"""G2 python_compat (M2a): precipitation typing & input vs RoadSurf-Python.

PrecPhase is missing (-9999) in the example, so the eq-42 sigmoid interpretation
path is exercised across the whole trajectory (incl. precipitation-active steps).
"""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.storage import calc_prec_type, precipitation_to_storage  # noqa: E402
from droad.ledger import make_ledger  # noqa: E402


def _cp_dict(cp):
    return {"MinPrecmm": cp.MinPrecmm, "MissValI": cp.MissValI,
            "PLimSnow": cp.PLimSnow, "PLimRain": cp.PLimRain}


@pytest.fixture(scope="module")
def prepared():
    sys.path.insert(0, str(RSP_SRC))
    return build_model()


def test_calc_prec_type_matches_reference_all_steps(prepared):
    m, objs = prepared
    modelInput, _, phy, ground, surf, atm, coupling, settings, condParam, _ = objs
    Cond = m["Cond"]
    n = settings.SimLen - 1
    cpd = _cp_dict(condParam)

    n_precip = 0
    for i in range(n):
        m["InputOutput"].SetCurrentValues(i, modelInput, atm, settings, surf, coupling, ground)
        ac = copy.deepcopy(atm)
        Cond.CalcPrecType(modelInput.PrecPhase[i], settings.DTSecs, ac, condParam)

        pt = calc_prec_type(atm.PrecInTStep, modelInput.PrecPhase[i],
                            atm.Tair, atm.Rhz, settings.DTSecs, cpd)

        assert pt.RainmmTS == pytest.approx(ac.RainmmTS, abs=1e-12, rel=0), f"step {i}"
        assert pt.SnowmmTS == pytest.approx(ac.SnowmmTS, abs=1e-12, rel=0), f"step {i}"
        assert pt.PrecType == ac.PrecType, f"step {i}"
        assert pt.RainIntensity == pytest.approx(ac.RainIntensity, abs=1e-12, rel=0)
        assert pt.SnowIntensity == pytest.approx(ac.SnowIntensity, abs=1e-12, rel=0)
        if pt.RainmmTS > 0 or pt.SnowmmTS > 0:
            n_precip += 1

    assert n_precip > 0  # sigmoid path actually exercised on precip steps


def test_precipitation_to_storage_conserves_and_adds():
    # rain adds to water, snow adds to snow; ledger residual 0
    from droad.storage import PrecType
    pt = PrecType(RainmmTS=0.3, SnowmmTS=0.0, PrecType=1, SnowType=2,
                  PrecInTStep=0.3, RainIntensity=36.0, SnowIntensity=0.0)
    wat, snow, lg = precipitation_to_storage(1.0, 0.0, 0.0, 0.0, pt)
    assert wat == pytest.approx(1.3)
    assert snow == 0.0
    assert lg.primary_mass_residual == pytest.approx(0.0, abs=1e-15)
    assert lg.external_source == pytest.approx(0.3)

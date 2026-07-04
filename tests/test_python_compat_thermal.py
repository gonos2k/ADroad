"""G0 python_compat: droad thermal kernel vs RoadSurf-Python.

Builds a real initialized state, then runs reference
(CalcHCapHCond -> calcCapDZCondDZ -> calcProfile) and the droad kernel on the
SAME inputs and compares the resulting temperature profile.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.thermal import calc_hcap_hcond, calc_cap_cond, calc_profile  # noqa: E402


@pytest.fixture(scope="module")
def state():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    return m, objs


def test_thermal_kernel_matches_reference(state):
    m, objs = state
    _, _, phy, ground, surf, atm, _, settings, _, _ = objs
    BalanceModel = m["BalanceModel"]

    NLayers, DTSecs = settings.NLayers, settings.DTSecs
    TrfFric, RNet = 10.0, 123.4          # fixed identical inputs for both sides
    atm.RNet = RNet

    # --- reference (mutates a deep copy) ---
    g = copy.deepcopy(ground)
    BalanceModel.CalcHCapHCond(NLayers, DTSecs, phy, g, atm)
    BalanceModel.calcCapDZCondDZ(NLayers, g)
    BalanceModel.calcProfile(NLayers, DTSecs, TrfFric, g, atm)
    ref_TmpNw = np.array(g.TmpNw, dtype=float)

    # --- droad (pure functions on the pristine ground arrays) ---
    phy_d = {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2}
    VSH, HS, GCond = calc_hcap_hcond(
        np.array(ground.TmpNw, float), np.array(ground.WCont, float),
        np.array(ground.CC, float), np.array(ground.ZDpth, float),
        NLayers, DTSecs, phy_d, atm.BLCond)
    condDZ, capDZ = calc_cap_cond(
        np.array(ground.CC, float), np.array(ground.DyK, float),
        np.array(ground.DyC, float), VSH, NLayers)
    TmpNw, _ = calc_profile(
        np.array(ground.Tmp, float), condDZ, capDZ, NLayers, DTSecs,
        TrfFric, atm.BLCond, RNet, atm.LE_Flux)

    assert np.allclose(TmpNw, ref_TmpNw, atol=1e-12, rtol=0)
    # surface temperature (avg of first two layers) parity
    assert (TmpNw[1] + TmpNw[2]) / 2 == pytest.approx(
        (ref_TmpNw[1] + ref_TmpNw[2]) / 2, abs=1e-12)


def test_hcap_intermediates_match_reference(state):
    m, objs = state
    _, _, phy, ground, surf, atm, _, settings, _, _ = objs
    NLayers, DTSecs = settings.NLayers, settings.DTSecs

    g = copy.deepcopy(ground)
    m["BalanceModel"].CalcHCapHCond(NLayers, DTSecs, phy, g, atm)

    phy_d = {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2}
    VSH, HS, GCond = calc_hcap_hcond(
        np.array(ground.TmpNw, float), np.array(ground.WCont, float),
        np.array(ground.CC, float), np.array(ground.ZDpth, float),
        NLayers, DTSecs, phy_d, atm.BLCond)

    assert np.allclose(VSH, np.array(g.VSH, float)[:NLayers], atol=1e-9, rtol=0)
    assert np.allclose(HS, np.array(g.HS, float)[:NLayers], atol=1e-9, rtol=0)

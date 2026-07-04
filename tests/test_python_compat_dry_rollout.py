"""G1b python_compat: free-running dry rollout vs reference dry rollout.

Both sides use the same driver semantics (SetCurrentValues obs-forcing +
one thermal step) with storage DISABLED, so this is a controlled dry experiment.
The reference oracle uses RoadSurf-Python physics; droad must track it over the
whole trajectory (error accumulation test), not just one step.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

from droad.driver import dry_rollout  # noqa: E402


def _phy_dict(phy):
    return {
        "Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
        "Emiss": phy.Emiss, "SB_const": phy.SB_const,
        "VK": phy.VK_Const, "logUstar": phy.logUstar, "logCond": phy.logCond,
        "logMom": phy.logMom, "logHeat": phy.logHeat,
        "ZRefT": phy.ZRefT, "Grav": phy.Grav, "LVap": phy.LVap, "LFus": phy.LFus,
    }


def _day_dict(s):
    return {
        "NightOn": s.NightOn, "NightOff": s.NightOff,
        "CalmLimDay": s.CalmLimDay, "CalmLimNgt": s.CalmLimNgt,
        "TrfFricDay": s.TrfFricDay, "TrfFricNgt": s.TrfFricNgt,
    }


def _reference_dry_trajectory(m, objs, n_steps):
    """Drive reference physics with storage disabled; record TsurfAve per step."""
    modelInput, _, phy, ground, surf, atm, coupling, settings, condParam, _ = objs
    out = np.empty(n_steps)
    for i in range(n_steps):
        m["InputOutput"].SetCurrentValues(i, modelInput, atm, settings, surf, coupling, ground)
        m["BalanceModel"].BalanceModelOneStep(
            modelInput.SW[i], modelInput.LW[i], phy, ground, surf, atm,
            settings, coupling, modelInput, i, condParam)
        out[i] = surf.TsurfAve
    return out


def test_dry_rollout_matches_reference():
    sys.path.insert(0, str(RSP_SRC))
    # --- reference oracle (fresh state) ---
    m, objs = build_model()
    modelInput, _, phy, ground, surf, atm, coupling, settings, condParam, _ = objs
    n_steps = min(400, settings.SimLen - 1)
    ref = _reference_dry_trajectory(m, objs, n_steps)

    # --- droad rollout (fresh state) ---
    m2, o2 = build_model()
    mi, _, phy2, g2, s2, a2, _, st2, _, _ = o2
    hours = np.array([t.hour for t in mi.time[:n_steps]], float)
    got = dry_rollout(
        Tair=np.array(mi.Tair, float), VZ=np.array(mi.VZ, float),
        Rhz=np.array(mi.Rhz, float), SW=np.array(mi.SW, float), LW=np.array(mi.LW, float),
        TSurfObs=np.array(mi.TSurfObs, float), hours=hours,
        Tmp0=g2.Tmp, TmpNw0=g2.TmpNw, WCont=np.array(g2.WCont, float),
        CC=np.array(g2.CC, float), ZDpth=np.array(g2.ZDpth, float),
        DyK=np.array(g2.DyK, float), DyC=np.array(g2.DyC, float),
        Albedo=g2.Albedo, BLCond0=a2.BLCond, TsurfAve0=s2.TsurfAve,
        NLayers=st2.NLayers, DTSecs=st2.DTSecs, InitLenI=st2.InitLenI,
        phy=_phy_dict(phy2), day=_day_dict(st2), n_steps=n_steps)

    rmse = float(np.sqrt(np.mean((got - ref) ** 2)))
    max_abs = float(np.max(np.abs(got - ref)))
    assert rmse <= 1e-6, f"dry rollout RMSE={rmse:.3e}, max={max_abs:.3e}"

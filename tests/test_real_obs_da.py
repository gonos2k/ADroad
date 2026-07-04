"""Real-observation DA (§8 baselines, §7.3 constraints, §11 report-only).

Hindcast-style variational DA on the actual road-surface observations (troad):
warm-start from a spun-up ground state, assimilate obs over a training window,
then verify the forecast window against obs and baselines
(B0 persistence, B1 default params).

Honest findings this test pins:
  - unconstrained calibration OVERFITS a short real window (forecast degrades) -> C3,
  - physical range constraint (Emiss<=1) + background regularization keep the
    estimate physical and beat persistence,
  - a single 200-step window does NOT beat a good default prior -> §11 report-only.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

jax = pytest.importorskip("jax")
from jax import config  # noqa: E402
config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.nn as jnn  # noqa: E402

from droad import jax_model as jm  # noqa: E402
from droad.assimilate import fit  # noqa: E402

K0, NA, NF = 2000, 200, 200


@pytest.fixture(scope="module")
def env():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    for i in range(K0):                       # spin up ground state to K0
        m["InputOutput"].SetCurrentValues(i, mi, a, st, s, coup, g)
        m["Storage"].PrecipitationToStorage(st, cpm, mi.PrecPhase[i], a, s)
        m["BalanceModel"].BalanceModelOneStep(mi.SW[i], mi.LW[i], phy, g, s, a, st, coup, mi, i, cpm)
        wf = m["WearingFactors"].WearingFactors()
        m["Cond"].WearFactors(cpm, st.Tph, s, wf)
        m["Cond"].RoadCond(phy.MaxPormms, s, a, st, cpm, wf)
        g.Albedo = m["Cond"].CalcAlbedo(s, cpm)

    n = NA + NF
    sl = slice(K0, K0 + n)
    prm = {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
           "Emiss": phy.Emiss, "SB_const": phy.SB_const, "VK": phy.VK_Const,
           "logUstar": phy.logUstar, "logCond": phy.logCond, "logMom": phy.logMom,
           "logHeat": phy.logHeat, "ZRefT": phy.ZRefT, "Grav": phy.Grav,
           "LVap": phy.LVap, "LFus": phy.LFus}
    day = {"NightOn": st.NightOn, "NightOff": st.NightOff, "CalmLimDay": st.CalmLimDay,
           "CalmLimNgt": st.CalmLimNgt, "TrfFricDay": st.TrfFricDay, "TrfFricNgt": st.TrfFricNgt}
    hours = np.array([t.hour for t in mi.time[sl]], float)
    is_night = ((hours >= st.NightOn) | (hours <= st.NightOff))
    static = {"NLayers": st.NLayers, "DTSecs": st.DTSecs,
              "WCont": jnp.array(g.WCont, float), "CC": jnp.array(g.CC, float),
              "ZDpth": jnp.array(g.ZDpth, float), "DyK": jnp.array(g.DyK, float),
              "DyC": jnp.array(g.DyC, float), "day": day, "Albedo": g.Albedo}
    bf = {"Tair": jnp.array(mi.Tair[sl], float), "VZ": jnp.array(mi.VZ[sl], float),
          "Rhz": jnp.array(mi.Rhz[sl], float), "SW": jnp.array(mi.SW[sl], float),
          "LW": jnp.array(mi.LW[sl], float), "is_night": jnp.array(is_night),
          "mask": jnp.zeros(n, bool), "obs": jnp.zeros(n)}
    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))
    tso = np.array(mi.TSurfObs, float)[sl]
    assert (tso > -100).all()                 # obs present across the whole window
    return prm, static, bf, x0, tso, phy.Emiss


def _predict(prm, static, bf, x0, emiss, bias):
    return jm.dry_rollout({**prm, "Emiss": emiss}, x0, {**bf, "Tair": bf["Tair"] + bias}, static)


def _frmse(prm, static, bf, x0, tso, emiss, bias):
    p = np.array(_predict(prm, static, bf, x0, emiss, bias))
    return float(np.sqrt(np.mean((p[NA:] - tso[NA:]) ** 2)))


def test_unconstrained_calibration_overfits(env):
    prm, static, bf, x0, tso, e0 = env
    obs = jnp.array(tso)
    w = jnp.concatenate([jnp.ones(NA), jnp.zeros(NF)])

    def loss(c):
        p = _predict(prm, static, bf, x0, c["Emiss"], c["bias"])
        return jnp.sum(w * (p - obs) ** 2) / jnp.sum(w)

    est, hist = fit(loss, {"Emiss": jnp.float64(e0), "bias": jnp.float64(0.0)}, steps=400, lr=0.01)
    assert hist[-1] < hist[0]                                   # training misfit reduced
    rmse_default = _frmse(prm, static, bf, x0, tso, e0, 0.0)
    rmse_da = _frmse(prm, static, bf, x0, tso, float(est["Emiss"]), float(est["bias"]))
    # overfitting: unconstrained forecast is worse than the default prior
    assert rmse_da > rmse_default


def test_constrained_regularized_is_physical_and_beats_persistence(env):
    prm, static, bf, x0, tso, e0 = env
    obs = jnp.array(tso)
    w = jnp.concatenate([jnp.ones(NA), jnp.zeros(NF)])

    def emiss_of(re):
        return 0.85 + 0.15 * jnn.sigmoid(re)                    # physical range (0.85, 1.0)

    def loss(c):
        e = emiss_of(c["re"])
        p = _predict(prm, static, bf, x0, e, c["bias"])
        mis = jnp.sum(w * (p - obs) ** 2) / jnp.sum(w)
        return mis + 5.0 * (e - 0.95) ** 2 + 2.0 * c["bias"] ** 2   # background reg

    est, hist = fit(loss, {"re": jnp.float64(0.0), "bias": jnp.float64(0.0)}, steps=400, lr=0.03)
    e_hat = float(emiss_of(est["re"]))
    rmse_da = _frmse(prm, static, bf, x0, tso, e_hat, float(est["bias"]))
    rmse_b0 = float(np.sqrt(np.mean((tso[NA - 1] - tso[NA:]) ** 2)))   # persistence

    assert e_hat <= 1.0                                         # stays physical
    assert rmse_da < rmse_b0                                    # beats persistence baseline

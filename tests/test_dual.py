"""Cycled dual estimation (§7.8): state (fast) + parameter (slow), twin.

A continuous truth is generated with a known Emiss; the estimator marches through
consecutive windows, each cycle doing a state analysis and one slow parameter
step, carrying the forecast as the next background. Asserts the robust behaviour
of a working dual system:
  - parameter tracks toward truth and settles (learning-rate decay),
  - window misfit drops across cycles,
  - state correction shrinks as the background improves.

Note: a small residual parameter bias is expected — state and parameter are
partly equifinal on a single station (the design's known caveat, §7.4/C3).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

jax = pytest.importorskip("jax")
from jax import config  # noqa: E402
config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from droad import jax_model as jm  # noqa: E402
from droad.dual import dual_estimation  # noqa: E402


def _base_static(objs):
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    base = {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
            "Emiss": phy.Emiss, "SB_const": phy.SB_const, "VK": phy.VK_Const,
            "logUstar": phy.logUstar, "logCond": phy.logCond, "logMom": phy.logMom,
            "logHeat": phy.logHeat, "ZRefT": phy.ZRefT, "Grav": phy.Grav,
            "LVap": phy.LVap, "LFus": phy.LFus}
    day = {"NightOn": st.NightOn, "NightOff": st.NightOff, "CalmLimDay": st.CalmLimDay,
           "CalmLimNgt": st.CalmLimNgt, "TrfFricDay": st.TrfFricDay, "TrfFricNgt": st.TrfFricNgt}
    static = {"NLayers": st.NLayers, "DTSecs": st.DTSecs,
              "WCont": jnp.array(g.WCont, float), "CC": jnp.array(g.CC, float),
              "ZDpth": jnp.array(g.ZDpth, float), "DyK": jnp.array(g.DyK, float),
              "DyC": jnp.array(g.DyC, float), "day": day, "Albedo": g.Albedo}
    return base, static


def test_cycled_dual_estimation_twin():
    m, objs = build_model()
    mi = objs[0]; st = objs[7]; g = objs[3]; a = objs[5]
    base, static = _base_static(objs)
    L, W, K0 = 70, 8, 1500

    def forc_slice(lo):
        sl = slice(lo, lo + L)
        hours = np.array([t.hour for t in mi.time[sl]], float)
        return {"Tair": jnp.array(mi.Tair[sl], float), "VZ": jnp.array(mi.VZ[sl], float),
                "Rhz": jnp.array(mi.Rhz[sl], float), "SW": jnp.array(mi.SW[sl], float),
                "LW": jnp.array(mi.LW[sl], float),
                "is_night": jnp.array((hours >= st.NightOn) | (hours <= st.NightOff)),
                "mask": jnp.zeros(L, bool), "obs": jnp.zeros(L)}

    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))

    def apply_state(bg, dx):
        return (bg[0].at[1:5].add(dx), bg[1].at[1:5].add(dx), bg[2])

    TRUE = 0.93
    windows, tbg = [], x0
    for wi in range(W):
        fc = forc_slice(K0 + wi * L)
        carry, ts = jm.dry_rollout_carry({**base, "Emiss": TRUE}, tbg, fc, static)
        windows.append((fc, ts)); tbg = carry

    bg0 = (x0[0].at[1:5].add(1.0), x0[1].at[1:5].add(1.0), x0[2])   # perturbed background
    sched = optax.exponential_decay(0.06, transition_steps=1, decay_rate=0.82)
    theta0 = 0.82
    theta, bg, hist = dual_estimation(base, theta0, bg0, windows, static, apply_state,
                                      state_dim=4, theta_lr=sched, state_steps=100,
                                      state_lr=0.05, bg_weight=0.02)

    th = hist["theta"]
    rmse = hist["window_rmse"]
    corr = hist["state_corr_norm"]

    # parameter tracked most of the way to truth and settled
    assert abs(th[-1] - TRUE) < abs(theta0 - TRUE) * 0.5
    assert abs(th[-1] - th[-2]) < 0.01
    # window misfit dropped substantially across cycles
    assert rmse[-1] < rmse[0] * 0.3
    # state correction shrank as the background improved (state tracking)
    assert corr[-1] < corr[0] * 0.5

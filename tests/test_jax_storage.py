"""M4 full (smooth_compat MVP): differentiable storage/phase-change rollout.

- dry reduction: with no precipitation, matches the dry JAX rollout.
- phase change active: synthetic precip + cold snap forms water then freezes to
  ice; gradients stay finite through the phase change.
- twin calibration through the wet model (Emiss recovered).
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

from droad import jax_model as jm, jax_storage as js  # noqa: E402
from droad.assimilate import fit  # noqa: E402


def _env(n=200, synthetic=False):
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    prm = {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
           "Emiss": phy.Emiss, "SB_const": phy.SB_const, "VK": phy.VK_Const,
           "logUstar": phy.logUstar, "logCond": phy.logCond, "logMom": phy.logMom,
           "logHeat": phy.logHeat, "ZRefT": phy.ZRefT, "Grav": phy.Grav,
           "LVap": phy.LVap, "LFus": phy.LFus, "PLimSnow": cpm.PLimSnow,
           "PLimRain": cpm.PLimRain, "TLimFreeze": cpm.TLimFreeze,
           "TLimMeltIce": cpm.TLimMeltIce, "AlbDry": cpm.AlbDry, "AlbSnow": cpm.AlbSnow,
           "MaxWatmms": cpm.MaxWatmms, "MaxIcemms": cpm.MaxIcemms,
           "MaxSnowmms": cpm.MaxSnowmms, "tau_T": 0.1, "tau_m": 0.01,
           "enth_L": 0.0, "enth_dT": 0.5}     # enthalpy off by default
    day = {"NightOn": st.NightOn, "NightOff": st.NightOff, "CalmLimDay": st.CalmLimDay,
           "CalmLimNgt": st.CalmLimNgt, "TrfFricDay": st.TrfFricDay, "TrfFricNgt": st.TrfFricNgt}
    hours = np.array([t.hour for t in mi.time[:n]], float)
    is_night = ((hours >= st.NightOn) | (hours <= st.NightOff))
    static = {"NLayers": st.NLayers, "DTSecs": st.DTSecs,
              "WCont": jnp.array(g.WCont, float), "CC": jnp.array(g.CC, float),
              "ZDpth": jnp.array(g.ZDpth, float), "DyK": jnp.array(g.DyK, float),
              "DyC": jnp.array(g.DyC, float), "day": day, "Albedo": g.Albedo}
    if synthetic:                       # cold snap + precip pulse (free-running twin)
        Tair = jnp.full(n, -5.0)
        Rhz = jnp.full(n, 90.0)
        SW = jnp.zeros(n)
        LW = jnp.full(n, 250.0)
        VZ = jnp.full(n, 3.0)
        prec = jnp.array(np.where((np.arange(n) >= 40) & (np.arange(n) < 70), 0.5, 0.0))
        mask = jnp.zeros(n, bool)
        obs = jnp.zeros(n)
    else:
        Tair = jnp.array(mi.Tair[:n], float); Rhz = jnp.array(mi.Rhz[:n], float)
        SW = jnp.array(mi.SW[:n], float); LW = jnp.array(mi.LW[:n], float)
        VZ = jnp.array(mi.VZ[:n], float)
        prec = jnp.array(np.array(mi.prec[:n], float) / 3600.0 * st.DTSecs)
        tso = np.array(mi.TSurfObs, float)[:n]
        mask = jnp.array((np.arange(n) <= st.InitLenI) & (tso > -100))
        obs = jnp.array(np.where(tso > -100, tso, 0.0))
    forc = {"Tair": Tair, "VZ": VZ, "Rhz": Rhz, "SW": SW, "LW": LW, "prec": prec,
            "is_night": jnp.array(is_night), "mask": mask, "obs": obs}
    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond),
          jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0), jnp.float64(cpm.AlbDry))
    return prm, static, forc, x0, n


def test_dry_reduction_matches_dry_model():
    prm, static, forc, x0, n = _env(300, synthetic=False)
    forc0 = {**forc, "prec": jnp.zeros(n)}
    wet0 = js.rollout(prm, x0, forc0, static)
    dforc = {k: forc[k] for k in ("Tair", "VZ", "Rhz", "SW", "LW", "is_night", "mask", "obs")}
    dry = jm.dry_rollout(prm, (x0[0], x0[1], x0[2]), dforc, static)
    assert float(jnp.max(jnp.abs(wet0 - dry))) < 1e-5


def _final_storage(prm, static, forc, x0):
    step = js.make_step(static)
    ft = (forc["Tair"], forc["VZ"], forc["Rhz"], forc["SW"], forc["LW"],
          forc["prec"], forc["is_night"], forc["mask"], forc["obs"])
    (_, _, _, W, S, I, _), _ = jax.lax.scan(lambda c, f: step(c, f, prm), x0, ft)
    return float(W), float(S), float(I)


def test_phase_change_activates_and_grad_finite():
    prm, static, forc, x0, n = _env(200, synthetic=True)
    W, S, I = _final_storage(prm, static, forc, x0)
    assert (S + I) > 0.05          # snow/ice actually formed (cold snap froze precip)

    w = jnp.ones(n)
    obs = js.rollout({**prm, "Emiss": 0.93}, x0, forc, obs_target := None) if False else \
        js.rollout({**prm, "Emiss": 0.93}, x0, forc, static)

    for key in ("Emiss", "AlbSnow", "TLimFreeze"):
        g = jax.grad(lambda v: js.loss({**prm, key: v}, x0, forc, obs, w, static))(prm[key])
        assert np.isfinite(float(g)), key


def test_twin_recover_emiss_through_wet_model():
    prm, static, forc, x0, n = _env(200, synthetic=True)
    w = jnp.ones(n)
    true = 0.93
    obs = js.rollout({**prm, "Emiss": true}, x0, forc, static)

    def loss(e):
        return js.loss({**prm, "Emiss": e}, x0, forc, obs, w, static)

    est, hist = fit(loss, jnp.float64(0.85), steps=300, lr=0.01)
    assert float(est) == pytest.approx(true, abs=2e-3)
    assert hist[-1] < hist[0] * 1e-4


def test_enhanced_enthalpy_mode(env=None):
    # enthalpy (§5a, N10): latent heat near 0C -> physics-changing mode.
    # enth_L=0 == base; enth_L>0 changes trajectory; grad wrt enth_L finite.
    prm, static, forc, x0, n = _env(200, synthetic=True)

    ts_off = js.rollout({**prm, "enth_L": 0.0}, x0, forc, static)
    ts_on = js.rollout({**prm, "enth_L": 3.0e8}, x0, forc, static)   # ~Lfus*rho_w
    assert float(jnp.max(jnp.abs(ts_on - ts_off))) > 1e-3            # physics changed

    w = jnp.ones(n)
    obs = js.rollout({**prm, "enth_L": 1.0e8}, x0, forc, static)
    g = jax.grad(lambda L: js.loss({**prm, "enth_L": L}, x0, forc, obs, w, static))(2.0e8)
    assert np.isfinite(float(g))                                     # differentiable

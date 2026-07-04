"""M3 + M4-dry: JAX dry rollout parity, differentiability, JVP/VJP dot-product.

- parity: JAX dry rollout (BLC-v0) == NumPy dry rollout (early_stop=False)
- gradient: jax.grad(loss) wrt Emiss and wrt initial profile x0, vs finite diff
- JVP<->VJP dot-product test on the rollout
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from droad import jax_model as jm  # noqa: E402
from droad.driver import dry_rollout as np_dry_rollout  # noqa: E402


def _phy(phy):
    return {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
            "Emiss": phy.Emiss, "SB_const": phy.SB_const, "VK": phy.VK_Const,
            "logUstar": phy.logUstar, "logCond": phy.logCond, "logMom": phy.logMom,
            "logHeat": phy.logHeat, "ZRefT": phy.ZRefT, "Grav": phy.Grav,
            "LVap": phy.LVap, "LFus": phy.LFus}


def _day(s):
    return {"NightOn": s.NightOn, "NightOff": s.NightOff, "CalmLimDay": s.CalmLimDay,
            "CalmLimNgt": s.CalmLimNgt, "TrfFricDay": s.TrfFricDay, "TrfFricNgt": s.TrfFricNgt}


def _setup(n=300):
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    phy_d, day_d = _phy(phy), _day(st)
    hours = np.array([t.hour for t in mi.time[:n]], float)
    is_night = ((hours >= st.NightOn) | (hours <= st.NightOff))
    tso = np.array(mi.TSurfObs, float)[:n]
    idx = np.arange(n)
    mask = (idx <= st.InitLenI) & (tso > -100.0)
    static = {
        "NLayers": st.NLayers, "DTSecs": st.DTSecs,
        "WCont": jnp.array(g.WCont, float), "CC": jnp.array(g.CC, float),
        "ZDpth": jnp.array(g.ZDpth, float), "DyK": jnp.array(g.DyK, float),
        "DyC": jnp.array(g.DyC, float), "day": day_d, "Albedo": g.Albedo,
    }
    forcings = {
        "Tair": jnp.array(mi.Tair[:n], float), "VZ": jnp.array(mi.VZ[:n], float),
        "Rhz": jnp.array(mi.Rhz[:n], float), "SW": jnp.array(mi.SW[:n], float),
        "LW": jnp.array(mi.LW[:n], float), "is_night": jnp.array(is_night),
        "mask": jnp.array(mask), "obs": jnp.array(np.where(tso > -100, tso, 0.0)),
    }
    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))
    return m, objs, phy_d, day_d, static, forcings, x0, hours, tso, mask, n


def test_jax_matches_numpy_dry_rollout():
    m, objs, phy_d, day_d, static, forcings, x0, hours, tso, mask, n = _setup(300)
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs

    jax_ts = np.array(jm.dry_rollout(phy_d, x0, forcings, static))

    np_ts = np_dry_rollout(
        Tair=np.array(mi.Tair, float), VZ=np.array(mi.VZ, float), Rhz=np.array(mi.Rhz, float),
        SW=np.array(mi.SW, float), LW=np.array(mi.LW, float),
        TSurfObs=np.array(mi.TSurfObs, float), hours=hours,
        Tmp0=g.Tmp, TmpNw0=g.TmpNw, WCont=np.array(g.WCont, float),
        CC=np.array(g.CC, float), ZDpth=np.array(g.ZDpth, float),
        DyK=np.array(g.DyK, float), DyC=np.array(g.DyC, float),
        Albedo=g.Albedo, BLCond0=a.BLCond, TsurfAve0=s.TsurfAve,
        NLayers=st.NLayers, DTSecs=st.DTSecs, InitLenI=st.InitLenI,
        phy=phy_d, day=day_d, n_steps=n, early_stop=False)   # BLC-v0 to match JAX

    assert np.allclose(jax_ts, np_ts, atol=1e-8, rtol=0), \
        f"max diff {np.max(np.abs(jax_ts - np_ts)):.3e}"


def test_grad_wrt_emiss_matches_finite_diff():
    m, objs, phy_d, day_d, static, forcings, x0, hours, tso, mask, n = _setup(200)
    obs = jnp.array(np.where(tso > -100, tso, 0.0))
    weight = jnp.array(mask, float)

    def loss_emiss(emiss):
        p = {**phy_d, "Emiss": emiss}
        return jm.loss(p, x0, forcings, obs, weight, static)

    g = float(jax.grad(loss_emiss)(phy_d["Emiss"]))
    h = 1e-6
    fd = (float(loss_emiss(phy_d["Emiss"] + h)) - float(loss_emiss(phy_d["Emiss"] - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-4), f"grad {g} vs fd {fd}"
    assert np.isfinite(g)


def test_grad_wrt_initial_profile_finite():
    # DA control: gradient wrt the initial temperature profile x0[0]
    m, objs, phy_d, day_d, static, forcings, x0, hours, tso, mask, n = _setup(150)
    obs = jnp.array(np.where(tso > -100, tso, 0.0))
    weight = jnp.array(mask, float)

    def loss_x0(Tmp0):
        return jm.loss(phy_d, (Tmp0, x0[1], x0[2]), forcings, obs, weight, static)

    g = jax.grad(loss_x0)(x0[0])
    assert np.all(np.isfinite(np.array(g)))
    assert np.any(np.array(g) != 0.0)


def test_jvp_vjp_dot_product():
    m, objs, phy_d, day_d, static, forcings, x0, hours, tso, mask, n = _setup(120)

    def f(emiss):
        p = {**phy_d, "Emiss": emiss}
        return jm.dry_rollout(p, x0, forcings, static)   # scalar -> vector

    e = phy_d["Emiss"]
    y, jv = jax.jvp(f, (e,), (1.0,))            # tangent (vector)
    _, vjp_fn = jax.vjp(f, e)
    u = jnp.ones_like(y)
    (jt_u,) = vjp_fn(u)                          # J^T u (scalar)
    lhs = float(jnp.vdot(jv, u))                 # <J v, u>
    rhs = float(1.0 * jt_u)                      # <v, J^T u>
    assert lhs == pytest.approx(rhs, rel=1e-8, abs=1e-8)

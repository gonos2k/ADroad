"""End-to-end demo: differentiable dROAD for data assimilation & calibration.

Runs two things and prints a summary:
  1) twin experiment  — recover a known Emiss + initial-state offset (synthetic)
  2) real-obs DA      — hindcast variational DA on the example troad observations,
                        naive vs constrained+regularized, vs baselines

    python examples/demo_da.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402
sys.path.insert(0, str(RSP_SRC))

from jax import config  # noqa: E402
config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp, jax.nn as jnn  # noqa: E402
from droad import jax_model as jm  # noqa: E402
from droad.assimilate import fit  # noqa: E402


def _phy(phy):
    return {"Poro1": phy.Poro1, "Poro2": phy.Poro2, "vsh1": phy.vsh1, "vsh2": phy.vsh2,
            "Emiss": phy.Emiss, "SB_const": phy.SB_const, "VK": phy.VK_Const,
            "logUstar": phy.logUstar, "logCond": phy.logCond, "logMom": phy.logMom,
            "logHeat": phy.logHeat, "ZRefT": phy.ZRefT, "Grav": phy.Grav,
            "LVap": phy.LVap, "LFus": phy.LFus}


def _day(s):
    return {"NightOn": s.NightOn, "NightOff": s.NightOff, "CalmLimDay": s.CalmLimDay,
            "CalmLimNgt": s.CalmLimNgt, "TrfFricDay": s.TrfFricDay, "TrfFricNgt": s.TrfFricNgt}


def _static_forc(mi, g, st, sl, mask):
    hours = np.array([t.hour for t in mi.time[sl]], float)
    static = {"NLayers": st.NLayers, "DTSecs": st.DTSecs,
              "WCont": jnp.array(g.WCont, float), "CC": jnp.array(g.CC, float),
              "ZDpth": jnp.array(g.ZDpth, float), "DyK": jnp.array(g.DyK, float),
              "DyC": jnp.array(g.DyC, float), "day": _day(st), "Albedo": g.Albedo}
    forc = {"Tair": jnp.array(mi.Tair[sl], float), "VZ": jnp.array(mi.VZ[sl], float),
            "Rhz": jnp.array(mi.Rhz[sl], float), "SW": jnp.array(mi.SW[sl], float),
            "LW": jnp.array(mi.LW[sl], float),
            "is_night": jnp.array((hours >= st.NightOn) | (hours <= st.NightOff)),
            "mask": jnp.array(mask), "obs": jnp.zeros(len(hours))}
    return static, forc


def demo_twin():
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    n = 200
    static, forc = _static_forc(mi, g, st, slice(0, n), np.zeros(n, bool))
    phy_d = _phy(phy)
    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))

    def off_x0(off):
        return (x0[0].at[1:5].add(off), x0[1].at[1:5].add(off), x0[2])

    true = {"Emiss": 0.93, "off": 2.0}
    obs = jm.dry_rollout({**phy_d, "Emiss": true["Emiss"]}, off_x0(true["off"]), forc, static)
    w = jnp.ones(n)

    def loss(c):
        return jm.loss({**phy_d, "Emiss": c["Emiss"]}, off_x0(c["off"]), forc, obs, w, static)

    est, hist = fit(loss, {"Emiss": jnp.float64(0.85), "off": jnp.float64(0.0)}, steps=500, lr=0.02)
    print("\n[1] Twin experiment (synthetic, joint param+state)")
    print(f"    true  Emiss={true['Emiss']:.3f}  offset={true['off']:.2f}")
    print(f"    est   Emiss={float(est['Emiss']):.5f}  offset={float(est['off']):.4f}")
    print(f"    loss  {hist[0]:.3e} -> {hist[-1]:.3e}")


def demo_real_obs(k0=2000, na=200, nf=200):
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    for i in range(k0):
        m["InputOutput"].SetCurrentValues(i, mi, a, st, s, coup, g)
        m["Storage"].PrecipitationToStorage(st, cpm, mi.PrecPhase[i], a, s)
        m["BalanceModel"].BalanceModelOneStep(mi.SW[i], mi.LW[i], phy, g, s, a, st, coup, mi, i, cpm)
        wf = m["WearingFactors"].WearingFactors(); m["Cond"].WearFactors(cpm, st.Tph, s, wf)
        m["Cond"].RoadCond(phy.MaxPormms, s, a, st, cpm, wf); g.Albedo = m["Cond"].CalcAlbedo(s, cpm)

    n = na + nf
    static, bf = _static_forc(mi, g, st, slice(k0, k0 + n), np.zeros(n, bool))
    phy_d = _phy(phy)
    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))
    tso = np.array(mi.TSurfObs, float)[k0:k0 + n]
    obs = jnp.array(tso); w = jnp.concatenate([jnp.ones(na), jnp.zeros(nf)])

    def predict(e, b):
        return jm.dry_rollout({**phy_d, "Emiss": e}, x0, {**bf, "Tair": bf["Tair"] + b}, static)

    def frmse(e, b):
        p = np.array(predict(e, b)); return float(np.sqrt(np.mean((p[na:] - tso[na:]) ** 2)))

    # naive
    est_n, _ = fit(lambda c: jnp.sum(w * (predict(c["Emiss"], c["bias"]) - obs) ** 2) / jnp.sum(w),
                   {"Emiss": jnp.float64(phy.Emiss), "bias": jnp.float64(0.0)}, steps=400, lr=0.01)
    # constrained + regularized
    emap = lambda re: 0.85 + 0.15 * jnn.sigmoid(re)

    def loss_reg(c):
        e = emap(c["re"]); p = predict(e, c["bias"])
        return jnp.sum(w * (p - obs) ** 2) / jnp.sum(w) + 5.0 * (e - 0.95) ** 2 + 2.0 * c["bias"] ** 2
    est_r, _ = fit(loss_reg, {"re": jnp.float64(0.0), "bias": jnp.float64(0.0)}, steps=400, lr=0.03)
    e_r = float(emap(est_r["re"]))

    b0 = float(np.sqrt(np.mean((tso[na - 1] - tso[na:]) ** 2)))
    print("\n[2] Real-observation DA (troad hindcast; forecast RMSE, degC)")
    print(f"    B0 persistence            {b0:.3f}")
    print(f"    B1 default (Emiss={phy.Emiss})    {frmse(phy.Emiss, 0.0):.3f}")
    print(f"    naive DA                  {frmse(float(est_n['Emiss']), float(est_n['bias'])):.3f}"
          f"   (Emiss={float(est_n['Emiss']):.3f}  <- overfit, unphysical)")
    print(f"    constrained+reg DA        {frmse(e_r, float(est_r['bias'])):.3f}"
          f"   (Emiss={e_r:.3f}  physical)")
    print("    -> single window does not beat a good prior (report-only, cf. §11)")


def demo_dual(W=8, L=70, k0=1500):
    from droad.dual import dual_estimation
    import optax
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    base = _phy(phy)

    static = {"NLayers": st.NLayers, "DTSecs": st.DTSecs,
              "WCont": jnp.array(g.WCont, float), "CC": jnp.array(g.CC, float),
              "ZDpth": jnp.array(g.ZDpth, float), "DyK": jnp.array(g.DyK, float),
              "DyC": jnp.array(g.DyC, float), "day": _day(st), "Albedo": g.Albedo}

    def fslice(lo):
        sl = slice(lo, lo + L)
        h = np.array([t.hour for t in mi.time[sl]], float)
        return {"Tair": jnp.array(mi.Tair[sl], float), "VZ": jnp.array(mi.VZ[sl], float),
                "Rhz": jnp.array(mi.Rhz[sl], float), "SW": jnp.array(mi.SW[sl], float),
                "LW": jnp.array(mi.LW[sl], float),
                "is_night": jnp.array((h >= st.NightOn) | (h <= st.NightOff)),
                "mask": jnp.zeros(L, bool), "obs": jnp.zeros(L)}

    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))
    apply_state = lambda bg, dx: (bg[0].at[1:5].add(dx), bg[1].at[1:5].add(dx), bg[2])
    TRUE, windows, tbg = 0.93, [], x0
    for wi in range(W):
        fc = fslice(k0 + wi * L)
        carry, ts = jm.dry_rollout_carry({**base, "Emiss": TRUE}, tbg, fc, static)
        windows.append((fc, ts)); tbg = carry
    bg0 = (x0[0].at[1:5].add(1.0), x0[1].at[1:5].add(1.0), x0[2])
    sched = optax.exponential_decay(0.06, transition_steps=1, decay_rate=0.82)
    theta, bg, hist = dual_estimation(base, 0.82, bg0, windows, static, apply_state,
                                      state_dim=4, theta_lr=sched, state_steps=100,
                                      state_lr=0.05, bg_weight=0.02)
    print("\n[3] Cycled dual estimation (state fast + parameter slow)")
    print(f"    true Emiss={TRUE}  start=0.82")
    print("    Emiss per cycle : " + " ".join(f"{t:.3f}" for t in hist["theta"]))
    print(f"    window RMSE     : {hist['window_rmse'][0]:.3f} -> {hist['window_rmse'][-1]:.3f}")
    print(f"    state corr norm : {hist['state_corr_norm'][0]:.2f} -> {hist['state_corr_norm'][-1]:.2f}")
    print("    -> state tracked each cycle; parameter converged/settled (small equifinality bias)")


if __name__ == "__main__":
    demo_twin()
    demo_real_obs()
    demo_dual()
    print("\nOK")

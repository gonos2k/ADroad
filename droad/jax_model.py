"""Differentiable dry thermal rollout in JAX (M3 + M4-dry).

Mirrors the NumPy dry path (droad.driver.dry_rollout) but with jnp + lax.scan,
so it is jit-able and differentiable. Domain-sensitive ops carry inline guards
(jnp.maximum) to keep gradients finite in the unselected branch — the JAX
counterpart of droad.branches (this file is the sanctioned jnp layer).

Differentiable inputs: physical params (e.g. Emiss) and the initial temperature
profile x0 — the control variables for calibration and variational DA.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import config, lax

config.update("jax_enable_x64", True)

_EPS = 1e-12


def _blc_v0(Tsurf, Tair, VZ, Rhz, BLCond0, SrfWat, dt, p, n_iter=40):
    TaK = Tair + 273.15
    AirDens = 100000.0 / (287.05 * TaK)
    AirHCap = 1005.0 + ((TaK - 250.0) ** 2) / 3364.0
    AirVCap = AirHCap * AirDens
    WatDen = -0.0050 * Tsurf * Tsurf + 0.0079 * Tsurf + 1000.0028

    def body(i, state):
        BLCond, PSIM, PSIH = state
        UStar = p["VK"] * VZ / (p["logUstar"] + PSIM)
        BLCond = AirVCap * p["VK"] * UStar / (p["logCond"] + PSIH)
        Stab = (-p["VK"] * p["ZRefT"] * p["Grav"] * BLCond * (Tsurf - Tair)
                / (AirVCap * (Tair + 273.15) * (UStar ** 3)))
        Stab = jnp.minimum(Stab, 1.0)
        # unstable branch guarded so its NaN never reaches the gradient
        arg = (1.0 + jnp.sqrt(jnp.maximum(1.0 - 16.0 * Stab, _EPS))) / 2.0
        psih_unstable = -2.0 * jnp.log(jnp.maximum(arg, _EPS))
        PSIH = jnp.where(Stab > 0, 4.7 * Stab, psih_unstable)
        PSIM = jnp.where(Stab > 0, PSIH, 0.6 * psih_unstable)
        return (BLCond, PSIM, PSIH)

    BLCond, PSIM, PSIH = lax.fori_loop(0, n_iter, body, (BLCond0, 0.0, 0.0))

    Raero = (p["logMom"] + PSIM) * (p["logHeat"] + PSIH) / (p["VK"] ** 2 * VZ)
    Raero = jnp.minimum(Raero, 30.0)

    PsychC = 0.1 * (0.00063 * TaK + 0.47496)
    esat_s = jnp.where(Tsurf < 0,
                       0.61078 * jnp.exp(21.875 * Tsurf / (Tsurf + 265.5)),
                       0.61078 * jnp.exp(17.269 * Tsurf / (Tsurf + 237.3)))
    esat_a = jnp.where(Tair < 0,
                       0.61078 * jnp.exp(21.875 * Tair / (Tair + 265.5)),
                       0.61078 * jnp.exp(17.269 * Tair / (Tair + 237.3)))
    EAir = jnp.minimum(0.01 * Rhz, 1.0) * esat_a
    LE = (AirDens * AirHCap * (esat_s - EAir)) / (PsychC * Raero)
    LE = jnp.where((LE > 0.0) & (SrfWat <= 0.0), 0.0, LE)   # no-water gate
    return BLCond, LE


def _calc_rnet(Emiss, SB, Tsurf, Albedo, SW, LW):
    TsK = Tsurf + 273.15
    TsK2 = TsK * TsK
    return (1.0 - Albedo) * SW + Emiss * LW - Emiss * SB * (TsK2 * TsK2)


def _hcap_vsh(TmpNw, WCont, phy, N):
    t = TmpNw[1:N + 1]
    t2 = t * t
    roo_w = -0.0050 * t2 + 0.0079 * t + 1000.0028
    cw_w = 0.0000102 * t2 * t2 - 0.0017169 * t2 * t + 0.11516 * t2 - 3.4739 * t + 4217.2
    roo = jnp.where(t >= 0, roo_w, 920.0)
    cw = jnp.where(t >= 0, cw_w, 2100.0)
    chwt = roo * cw
    dry = jnp.where(jnp.arange(N) <= 1,
                    (1.0 - phy["Poro1"]) * phy["vsh1"],
                    (1.0 - phy["Poro2"]) * phy["vsh2"])
    return dry + WCont * chwt


def make_dry_step(static):
    N, dt = static["NLayers"], static["DTSecs"]
    WCont, CC, ZDpth = static["WCont"], static["CC"], static["ZDpth"]
    DyK, DyC = static["DyK"], static["DyC"]
    day = static["day"]
    Albedo = static["Albedo"]

    def step(carry, forc, params):
        Tmp, TmpNw, BLCond = carry
        Tair, VZ, Rhz, SW, LW, is_night, mask, obs = forc

        Tmp = Tmp.at[0].set(Tair)
        Tmp = Tmp.at[1].set(jnp.where(mask, obs, Tmp[1]))
        Tmp = Tmp.at[2].set(jnp.where(mask, obs, Tmp[2]))
        Tsurf = (Tmp[1] + Tmp[2]) / 2.0

        CalmLim = jnp.where(is_night, day["CalmLimNgt"], day["CalmLimDay"])
        TrfFric = jnp.where(is_night, day["TrfFricNgt"], day["TrfFricDay"])
        VZc = jnp.maximum(VZ, CalmLim)

        BLCond, LE = _blc_v0(Tsurf, Tair, VZc, Rhz, BLCond, 0.0, dt, params)
        RNet = _calc_rnet(params["Emiss"], params["SB_const"], Tsurf, Albedo, SW, LW)

        VSH = _hcap_vsh(TmpNw, WCont, params, N)
        condDZ = -(CC / DyK[:N])
        capDZ = -(1.0 / (DyC[:N] * VSH))

        GFlux0 = RNet - LE + TrfFric + BLCond * (Tmp[0] - Tmp[1])
        GfluxJ = condDZ * (Tmp[2:N + 2] - Tmp[1:N + 1])
        Gflux = jnp.concatenate([jnp.array([GFlux0]), GfluxJ])   # length N+1
        TmpNw_new = Tmp.copy()
        upd = Tmp[1:N + 1] + dt * capDZ * (Gflux[1:] - Gflux[:-1])
        TmpNw_new = TmpNw_new.at[1:N + 1].set(upd)

        Tsurf_new = (TmpNw_new[1] + TmpNw_new[2]) / 2.0
        return (TmpNw_new, TmpNw_new, BLCond), Tsurf_new

    return step


def dry_rollout(params, x0, forcings, static):
    """params: dict (Emiss, SB_const, VK, logs..., Poro/vsh...).
    x0: (Tmp0, TmpNw0, BLCond0). forcings: dict of (T,) arrays. Returns Tsurf (T,)."""
    step = make_dry_step(static)
    forc = (forcings["Tair"], forcings["VZ"], forcings["Rhz"], forcings["SW"],
            forcings["LW"], forcings["is_night"], forcings["mask"], forcings["obs"])
    scan_forc = tuple(f for f in forc)

    def scan_step(carry, f):
        return step(carry, f, params)

    carry0 = (x0[0], x0[1], x0[2])
    _, tsurf = lax.scan(scan_step, carry0, scan_forc)
    return tsurf


def dry_rollout_carry(params, x0, forcings, static):
    """Like dry_rollout but also returns the final carry (Tmp, TmpNw, BLCond),
    i.e. the forecast end-state — used as the next cycle's background (cycling)."""
    step = make_dry_step(static)
    forc = (forcings["Tair"], forcings["VZ"], forcings["Rhz"], forcings["SW"],
            forcings["LW"], forcings["is_night"], forcings["mask"], forcings["obs"])
    carry, tsurf = lax.scan(lambda c, f: step(c, f, params), (x0[0], x0[1], x0[2]), forc)
    return carry, tsurf


def loss(params, x0, forcings, obs_tsurf, weight, static):
    """Weighted MSE of surface temperature (control = params and/or x0)."""
    pred = dry_rollout(params, x0, forcings, static)
    return jnp.sum(weight * (pred - obs_tsurf) ** 2) / jnp.sum(weight)

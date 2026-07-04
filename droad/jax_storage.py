"""smooth_compat JAX rollout with storage & phase change (M4 full, MVP).

Extends the dry JAX model with a differentiable storage path:
  precipitation split (eq-42 sigmoid + smooth rain/snow ramp) -> water/snow,
  smooth freeze (water->ice) and melt (ice/snow->water) gates,
  smooth albedo (dry<->snow). Non-negativity/overflow use jnp.clip (subgrad OK);
  phase gates use smoothing.gate so gradients flow across thresholds.

This is a smooth_compat MVP (not bit-exact to the hard model). Precisely: it uses
smooth phase gates and precipitation ramps, but keeps HARD non-negativity /
capacity projections (jnp.clip) on the water/ice/snow stores to preserve the
dry-reduction and non-negative-mass invariants. So it is differentiable almost
everywhere with a subgradient at the storage bounds — not "smooth everywhere".
It reduces to the dry rollout when there is no precipitation / phase activity
(deviation-budget territory, not python_compat bit parity).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import config, lax

from . import smoothing as sm
from .jax_model import _blc_v0, _calc_rnet, _hcap_vsh

config.update("jax_enable_x64", True)


def make_step(static):
    N, dt = static["NLayers"], static["DTSecs"]
    WCont, CC, ZDpth = static["WCont"], static["CC"], static["ZDpth"]
    DyK, DyC, day = static["DyK"], static["DyC"], static["day"]

    def step(carry, forc, prm):
        Tmp, TmpNw, BLCond, Wat, Snow, Ice, Alb = carry
        Tair, VZ, Rhz, SW, LW, prec, is_night, mask, obs = forc
        tau_T = prm["tau_T"]     # temperature-gate width (deg C)
        tau_m = prm["tau_m"]     # mass-gate width (mm)

        # obs forcing (init window)
        Tmp = Tmp.at[0].set(Tair)
        Tmp = Tmp.at[1].set(jnp.where(mask, obs, Tmp[1]))
        Tmp = Tmp.at[2].set(jnp.where(mask, obs, Tmp[2]))
        Tsurf = (Tmp[1] + Tmp[2]) / 2.0

        # precipitation split (eq 42) -> smooth rain fraction over [PLimSnow, PLimRain]
        p_exp = jnp.clip(22.0 - 2.7 * Tair - 0.20 * Rhz, -60.0, 60.0)
        p_rain = 1.0 / (1.0 + jnp.exp(p_exp))
        # guard the ramp width: PLimRain must stay above PLimSnow even if a DA
        # control nudges the thresholds together / past each other (else NaN or a
        # sign-flipped rain/snow split).
        den = jnp.maximum(prm["PLimRain"] - prm["PLimSnow"], 1e-4)
        wfrac = sm.soft_clip((p_rain - prm["PLimSnow"]) / den, 0.0, 1.0, 0.05)
        Wat = Wat + prec * wfrac
        Snow = Snow + prec * (1.0 - wfrac)

        # balance (uses carried albedo, as in the reference order)
        CalmLim = jnp.where(is_night, day["CalmLimNgt"], day["CalmLimDay"])
        TrfFric = jnp.where(is_night, day["TrfFricNgt"], day["TrfFricDay"])
        VZc = jnp.maximum(VZ, CalmLim)
        BLCond, LE = _blc_v0(Tsurf, Tair, VZc, Rhz, BLCond, Wat, dt, prm)
        RNet = _calc_rnet(prm["Emiss"], prm["SB_const"], Tsurf, Alb, SW, LW)

        VSH = _hcap_vsh(TmpNw, WCont, prm, N)
        # enhanced_enthalpy (§5a, N10): latent heat absorbed near 0 C via apparent
        # heat capacity -> damps the freeze/melt temperature bounce. enth_L=0 -> off.
        # enth_dT floored (same guard as gate) so the 1/dT latent term stays finite.
        enth_dT = sm.safe_tau(prm["enth_dT"])
        s = sm.gate(TmpNw[1:N + 1], 0.0, enth_dT)
        VSH = VSH + prm["enth_L"] * WCont * (s * (1.0 - s) / enth_dT)
        condDZ = -(CC / DyK[:N])
        capDZ = -(1.0 / (DyC[:N] * VSH))
        GFlux0 = RNet - LE + TrfFric + BLCond * (Tmp[0] - Tmp[1])
        GfluxJ = condDZ * (Tmp[2:N + 2] - Tmp[1:N + 1])
        Gflux = jnp.concatenate([jnp.array([GFlux0]), GfluxJ])
        TmpNw_new = Tmp.copy()
        TmpNw_new = TmpNw_new.at[1:N + 1].set(
            Tmp[1:N + 1] + dt * capDZ * (Gflux[1:] - Gflux[:-1]))
        Tsurf_new = (TmpNw_new[1] + TmpNw_new[2]) / 2.0

        # smooth phase change (temperature gates)
        freeze_g = 1.0 - sm.gate(Tsurf_new, prm["TLimFreeze"], tau_T)   # ~1 when cold
        mfr = sm.transfer(Wat, freeze_g)                                # water -> ice
        Wat, Ice = Wat - mfr, Ice + mfr

        melt_g = sm.gate(Tsurf_new, prm["TLimMeltIce"], tau_T)          # ~1 when warm
        mmi = sm.transfer(Ice, melt_g)                                 # ice -> water
        Ice, Wat = Ice - mmi, Wat + mmi
        mms = sm.transfer(Snow, melt_g)                               # snow -> water
        Snow, Wat = Snow - mms, Wat + mms

        # non-negativity / overflow: hard clip (subgradient OK). A soft clamp at
        # the 0 floor would add ~tau*log2 spurious mass even when dry, breaking
        # the dry-reduction invariant, so the lower bound stays a hard projection.
        Wat = jnp.clip(Wat, 0.0, prm["MaxWatmms"])
        Ice = jnp.clip(Ice, 0.0, prm["MaxIcemms"])
        Snow = jnp.clip(Snow, 0.0, prm["MaxSnowmms"])

        # smooth albedo (dry -> snow), mass-gate width so Snow=0 -> AlbDry
        Alb = sm.select(Snow, 0.1, tau_m, prm["AlbSnow"], prm["AlbDry"])

        return (TmpNw_new, TmpNw_new, BLCond, Wat, Snow, Ice, Alb), Tsurf_new

    return step


def rollout(params, x0, forcings, static):
    """x0=(Tmp0,TmpNw0,BLCond0,Wat0,Snow0,Ice0,Alb0). Returns Tsurf (T,)."""
    step = make_step(static)
    forc = (forcings["Tair"], forcings["VZ"], forcings["Rhz"], forcings["SW"],
            forcings["LW"], forcings["prec"], forcings["is_night"],
            forcings["mask"], forcings["obs"])

    def scan_step(carry, f):
        return step(carry, f, params)

    _, tsurf = lax.scan(scan_step, x0, forc)
    return tsurf


def loss(params, x0, forcings, obs_tsurf, weight, static):
    pred = rollout(params, x0, forcings, static)
    return jnp.sum(weight * (pred - obs_tsurf) ** 2) / jnp.sum(weight)

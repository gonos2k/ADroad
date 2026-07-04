"""Cycled dual estimation — state (fast) + parameter (slow) (§7.8).

Elevates dROAD from batch joint optimization to a **cycling hybrid DA system**:
march through consecutive assimilation windows and, in each cycle,
  1) STATE analysis  — optimize this window's initial state given fixed params
                       and a background from the previous forecast (variational),
  2) PARAM update    — one slow gradient step on the shared global parameter,
  3) FORECAST        — run the window to its end; that end-state is the next
                       cycle's background (state carried across cycles).

The two estimators run on different timescales (state per-cycle, parameter across
cycles) — the "dual" of dual estimation. The low-dimensional state control +
background regularization keep the parameter identifiable (state cannot fully
absorb parameter error).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax

from . import jax_model as jm
from .assimilate import fit


def _make_theta_map(bounds):
    """Map an unconstrained z to theta. If bounds=(lo,hi), keep theta in that
    physical range via a sigmoid (and invert to initialize z from theta0)."""
    if bounds is None:
        return (lambda z: z), (lambda t: jnp.asarray(t, float))
    lo, hi = bounds

    def to_theta(z):
        return lo + (hi - lo) * jax.nn.sigmoid(z)

    def to_z(t):
        # one-time scalar init on host: invert the sigmoid (logit), no jnp prims
        p = min(max((float(t) - lo) / (hi - lo), 1e-6), 1.0 - 1e-6)
        return jnp.asarray(math.log(p / (1.0 - p)), float)

    return to_theta, to_z


def dual_estimation(base, theta0, bg0, windows, static, apply_state, *,
                    state_dim=4, theta_lr=0.05, state_steps=120, state_lr=0.05,
                    bg_weight=1e-3, theta_key="Emiss", bounds=None):
    """Run cycled dual estimation over `windows` = list of (forcings, obs).

    `base`     : fixed physical params dict (theta_key is overwritten by theta).
    `bg0`      : initial background carry (Tmp, TmpNw, BLCond).
    `apply_state(bg, dx)` : produce an initial state (x0 tuple) from background +
                            a `state_dim`-vector correction dx.
    `bounds`   : optional (lo, hi); keeps theta physical via sigmoid reparam.

    Returns (theta, bg, history). history keys, one entry per cycle:
      theta_before / theta_after : parameter before and after that cycle's update
      window_rmse                : misfit measured with theta_before (pre-update)
      state_corr_norm            : ||state correction|| for the cycle
      theta                      : alias of theta_after (post-update trace)
    """
    to_theta, to_z = _make_theta_map(bounds)
    z = to_z(theta0)                              # optimize in z-space
    bg = bg0
    topt = optax.adam(theta_lr)
    tstate = topt.init(z)
    hist = {"theta": [], "theta_before": [], "theta_after": [],
            "state_corr_norm": [], "window_rmse": []}

    for forc, obs in windows:
        wgt = jnp.ones_like(obs)
        theta_before = float(to_theta(z))

        # (1) STATE analysis: fit low-dim state correction, params fixed
        def sloss(dx):
            x0 = apply_state(bg, dx)
            pred = jm.dry_rollout({**base, theta_key: to_theta(z)}, x0, forc, static)
            return (jnp.sum(wgt * (pred - obs) ** 2) / jnp.sum(wgt)
                    + bg_weight * jnp.sum(dx ** 2))

        dx, _ = fit(sloss, jnp.zeros(state_dim), steps=state_steps, lr=state_lr)
        x0a = apply_state(bg, dx)

        # (2) PARAM update: one slow gradient step, state fixed at analysis
        def zloss(zz):
            pred = jm.dry_rollout({**base, theta_key: to_theta(zz)}, x0a, forc, static)
            return jnp.mean((pred - obs) ** 2)

        rmse_before = float(zloss(z)) ** 0.5      # misfit at theta_before
        g = jax.grad(zloss)(z)
        upd, tstate = topt.update(g, tstate, z)
        z = optax.apply_updates(z, upd)
        theta_after = float(to_theta(z))

        # (3) FORECAST: end-state becomes next cycle's background
        bg, _ = jm.dry_rollout_carry({**base, theta_key: to_theta(z)}, x0a, forc, static)

        hist["theta_before"].append(theta_before)
        hist["theta_after"].append(theta_after)
        hist["theta"].append(theta_after)
        hist["state_corr_norm"].append(float(jnp.linalg.norm(dx)))
        hist["window_rmse"].append(rmse_before)

    return to_theta(z), bg, hist

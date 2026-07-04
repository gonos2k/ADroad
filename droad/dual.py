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

import jax
import jax.numpy as jnp
import optax

from . import jax_model as jm
from .assimilate import fit


def dual_estimation(base, theta0, bg0, windows, static, apply_state, *,
                    state_dim=4, theta_lr=0.05, state_steps=120, state_lr=0.05,
                    bg_weight=1e-3, theta_key="Emiss"):
    """Run cycled dual estimation over `windows` = list of (forcings, obs).

    `base`     : fixed physical params dict (theta_key is overwritten by theta).
    `bg0`      : initial background carry (Tmp, TmpNw, BLCond).
    `apply_state(bg, dx)` : produce an initial state (x0 tuple) from background +
                            a `state_dim`-vector correction dx.
    Returns (theta, bg, history).
    """
    theta = jnp.asarray(theta0, float)
    bg = bg0
    topt = optax.adam(theta_lr)
    tstate = topt.init(theta)
    hist = {"theta": [], "state_corr_norm": [], "window_rmse": []}

    for forc, obs in windows:
        wgt = jnp.ones_like(obs)

        # (1) STATE analysis: fit low-dim state correction, params fixed
        def sloss(dx):
            x0 = apply_state(bg, dx)
            pred = jm.dry_rollout({**base, theta_key: theta}, x0, forc, static)
            return (jnp.sum(wgt * (pred - obs) ** 2) / jnp.sum(wgt)
                    + bg_weight * jnp.sum(dx ** 2))

        dx, _ = fit(sloss, jnp.zeros(state_dim), steps=state_steps, lr=state_lr)
        x0a = apply_state(bg, dx)

        # (2) PARAM update: one slow gradient step, state fixed at analysis
        def tloss(th):
            pred = jm.dry_rollout({**base, theta_key: th}, x0a, forc, static)
            return jnp.mean((pred - obs) ** 2)

        rmse = float(tloss(theta)) ** 0.5
        g = jax.grad(tloss)(theta)
        upd, tstate = topt.update(g, tstate, theta)
        theta = optax.apply_updates(theta, upd)

        # (3) FORECAST: end-state becomes next cycle's background
        bg, _ = jm.dry_rollout_carry({**base, theta_key: theta}, x0a, forc, static)

        hist["theta"].append(float(theta))
        hist["state_corr_norm"].append(float(jnp.linalg.norm(dx)))
        hist["window_rmse"].append(rmse)

    return theta, bg, hist

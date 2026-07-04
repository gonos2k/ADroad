"""Smooth surrogates for hard branches / clamps / phase change (§5, smooth_compat).

Each surrogate has a width tau (or dT) and converges to its hard counterpart as
tau -> 0, while staying differentiable at the threshold. Used to build the
`smooth_compat` model mode. JAX layer (jnp), so gradients flow.

Contract:
  gate(x, thr, tau)           -> ~0 below thr, ~1 above; d/dx finite everywhere
  select(x, thr, tau, hi, lo) -> hi where x>thr else lo (smooth)
  soft_min(x, cap, tau)       -> min(x, cap)
  soft_max(x, flr, tau)       -> max(x, flr)
  transfer(available, g)      -> mass moved = g*available (0<=g<=1) -> conserving
  ceff(T, c, Lfus, dT)        -> effective heat capacity absorbing latent heat
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp

_TAU_MIN = 1e-6      # floor on gate/clamp widths (tau may be a DA control)
_Z_CLIP = 60.0       # bound the sigmoid/softplus argument (avoid exp overflow)


def safe_tau(tau):
    """Floor a smoothing width so 1/tau denominators can't blow up (tau may be a
    DA control that wanders to 0 or negative)."""
    return jnp.maximum(tau, _TAU_MIN)


_safe_tau = safe_tau  # internal alias


def gate(x, thr, tau):
    """Smooth step in [0,1]. Guards tau>0 and clips the argument (no overflow)."""
    z = jnp.clip((x - thr) / _safe_tau(tau), -_Z_CLIP, _Z_CLIP)
    return jnn.sigmoid(z)


def select(x, thr, tau, hi, lo):
    g = gate(x, thr, tau)
    return g * hi + (1.0 - g) * lo


def soft_min(x, cap, tau):
    # cap - softplus(cap - x). logaddexp is numerically stable (no overflow) and
    # its linear regime is what makes tau->0 converge to hard min, so the
    # argument must NOT be clipped here. Only tau is floored.
    tau = _safe_tau(tau)
    return cap - tau * jnp.logaddexp(0.0, (cap - x) / tau)


def soft_max(x, flr, tau):
    tau = _safe_tau(tau)
    return flr + tau * jnp.logaddexp(0.0, (x - flr) / tau)


def soft_clip(x, lo, hi, tau):
    """Differentiable clamp to [lo, hi] (soft_max then soft_min)."""
    return soft_min(soft_max(x, lo, tau), hi, tau)


def transfer(available, g):
    """Mass moved = g * available (g in [0,1]); structurally conserving/non-neg."""
    return g * available


def melt_fraction(T, Tmelt, dT):
    """Smooth melted fraction in [0,1], 0.5 at Tmelt."""
    return gate(T, Tmelt, dT)


def ceff(T, c, Lfus, Tmelt, dT):
    """Effective heat capacity: base c plus latent heat spread over width dT
    around Tmelt (apparent-heat-capacity method). Integral over T recovers
    c*T + Lfus*melt_fraction, so latent heat is energy-conserving. dT is floored
    so the latent term can't explode/flip sign as dT -> 0 (gate already floors it,
    but the 1/dT factor here needs the same guard)."""
    dT = safe_tau(dT)
    s = gate(T, Tmelt, dT)
    dphi = s * (1.0 - s) / dT          # d(melt_fraction)/dT
    return c + Lfus * dphi

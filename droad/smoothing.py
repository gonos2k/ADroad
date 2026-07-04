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

import jax.numpy as jnp


def gate(x, thr, tau):
    return 1.0 / (1.0 + jnp.exp(-(x - thr) / tau))


def select(x, thr, tau, hi, lo):
    g = gate(x, thr, tau)
    return g * hi + (1.0 - g) * lo


def soft_min(x, cap, tau):
    # cap - softplus(cap - x)  (tau-scaled)
    return cap - tau * jnp.logaddexp(0.0, (cap - x) / tau)


def soft_max(x, flr, tau):
    return flr + tau * jnp.logaddexp(0.0, (x - flr) / tau)


def transfer(available, g):
    """Mass moved = g * available (g in [0,1]); structurally conserving/non-neg."""
    return g * available


def melt_fraction(T, Tmelt, dT):
    """Smooth melted fraction in [0,1], 0.5 at Tmelt."""
    return gate(T, Tmelt, dT)


def ceff(T, c, Lfus, Tmelt, dT):
    """Effective heat capacity: base c plus latent heat spread over width dT
    around Tmelt (apparent-heat-capacity method). Integral over T recovers
    c*T + Lfus*melt_fraction, so latent heat is energy-conserving."""
    s = gate(T, Tmelt, dT)
    dphi = s * (1.0 - s) / dT          # d(melt_fraction)/dT
    return c + Lfus * dphi

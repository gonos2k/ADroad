"""Branch & domain-guard wrappers (P0 §4).

Rule: core code MUST NOT call raw `where`, `cond`, `sqrt`, `log`, `exp`, `clip`.
It must go through a site-aware wrapper here, and the site must be listed in
BRANCH_REGISTRY. A CI audit (tools/check_raw_primitives.py) enforces the ban.

Backend-neutral: uses NumPy now (M1). The same wrappers wrap jnp later (M2+),
so the registry/contract is unchanged when the backend switches.
"""

from __future__ import annotations

import numpy as np

# site -> policy. One entry per data-dependent branch / domain-guarded op in core.
BRANCH_REGISTRY: dict[str, str] = {
    # boundary layer
    "boundary_layer.psi_unstable": "guarded_math",   # sqrt(1 - 16*zeta), log(...)
    "boundary_layer.sat_vapor": "guarded_math",      # exp() in saturation vapor pressure
    "boundary_layer.raero_cap": "safe_where",
    "calc_le.no_water_gate": "custom_smooth",
    # heat / ground
    "heat_capacity.water_ice_props": "safe_where",
    # storage
    "precip.p_rain_sigmoid": "guarded_math",         # 1/(1+exp(p_exp)), eq 42
    "storage.freeze_gate": "custom_smooth",
    "storage.melt_gate": "custom_smooth",
    "storage.min_clip": "safe_where",
    "storage.max_clip": "safe_where",
    "wearfactors.snow_lt_0p2": "custom_smooth",
    # albedo
    "albedo.snow_ice_switch": "custom_smooth",
}

_VALID_POLICIES = {"safe_where", "guarded_where", "guarded_math", "custom_smooth", "lax_cond"}


class BranchError(ValueError):
    """Raised when a branch site is unregistered or has the wrong policy."""


def assert_branch_registered(site: str, policy: str) -> None:
    if policy not in _VALID_POLICIES:
        raise BranchError(f"unknown branch policy {policy!r}")
    have = BRANCH_REGISTRY.get(site)
    if have is None:
        raise BranchError(f"branch site {site!r} is not registered")
    if have != policy:
        raise BranchError(f"site {site!r} registered as {have!r}, called as {policy!r}")


def safe_where(site, cond, x, y):
    """Select between two branches that are BOTH valid for all inputs."""
    assert_branch_registered(site, "safe_where")
    return np.where(cond, x, y)


def guarded_where(site, cond, x, y):
    """Select where a branch needs sanitized inputs (caller pre-guards)."""
    assert_branch_registered(site, "guarded_where")
    return np.where(cond, x, y)


def guarded_sqrt(site, x, eps=1e-12):
    """sqrt with domain guard: sqrt(max(x, eps))."""
    assert_branch_registered(site, "guarded_math")
    return np.sqrt(np.maximum(x, eps))


def guarded_log(site, x, eps=1e-12):
    assert_branch_registered(site, "guarded_math")
    return np.log(np.maximum(x, eps))


def guarded_exp(site, x, lo=-60.0, hi=60.0):
    """exp routed through a registered site; argument clipped to avoid overflow.

    Parity caveat: the reference does NOT clip. Clipping only changes the result
    once |x|>~60 (exp already 0 or ~1e26), which is outside the physical/fixture
    domain, so bit-exact parity holds on the fixture but not for arbitrary extreme
    inputs. Documented in README (exact parity = normal-range claim)."""
    assert_branch_registered(site, "guarded_math")
    return np.exp(np.clip(x, lo, hi))

"""M4 smooth_compat primitives: tau->0 convergence + differentiability (§5, P0).

Convergence is checked at the PRIMITIVE level (per P0): each surrogate -> its
hard counterpart as tau shrinks, and stays finite-grad at the threshold.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from jax import config  # noqa: E402
config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from droad import smoothing as sm  # noqa: E402


XS = jnp.linspace(-5.0, 5.0, 201)


def test_gate_converges_to_step():
    err = []
    far = jnp.abs(XS) > 1.0          # fixed region (independent of tau)
    for tau in (0.5, 0.1, 0.02):
        hard = jnp.where(XS > 0.0, 1.0, 0.0)
        soft = sm.gate(XS, 0.0, tau)
        err.append(float(jnp.max(jnp.abs(soft - hard)[far])))
    assert err[0] > err[1] > err[2]
    assert err[-1] < 1e-3


def test_select_converges_to_where():
    hi, lo = 2.0, -1.0
    hard = jnp.where(XS > 0.5, hi, lo)
    soft = sm.select(XS, 0.5, 0.01, hi, lo)
    far = jnp.abs(XS - 0.5) > 0.1
    assert float(jnp.max(jnp.abs(soft - hard)[far])) < 1e-3


def test_soft_min_max_converge():
    assert np.allclose(np.array(sm.soft_min(XS, 1.0, 0.01)),
                       np.array(jnp.minimum(XS, 1.0)), atol=5e-2)
    assert np.allclose(np.array(sm.soft_max(XS, -1.0, 0.001)),
                       np.array(jnp.maximum(XS, -1.0)), atol=1e-2)
    # tighter tau -> tighter bound
    e1 = float(jnp.max(jnp.abs(sm.soft_min(XS, 1.0, 0.1) - jnp.minimum(XS, 1.0))))
    e2 = float(jnp.max(jnp.abs(sm.soft_min(XS, 1.0, 0.01) - jnp.minimum(XS, 1.0))))
    assert e2 < e1


def test_grad_finite_at_threshold():
    # the whole point: differentiable exactly where the hard branch is not
    for fn in (lambda x: sm.gate(x, 0.0, 0.1),
               lambda x: sm.soft_min(x, 0.0, 0.1),
               lambda x: sm.soft_max(x, 0.0, 0.1)):
        g = float(jax.grad(fn)(0.0))
        assert np.isfinite(g)


def test_transfer_conserves():
    # moving g*available out of A into B conserves A+B and stays non-negative
    A, B, g = 2.0, 1.0, sm.gate(jnp.float64(1.0), 0.0, 0.1)
    moved = sm.transfer(A, g)
    A2, B2 = A - moved, B + moved
    assert float(A2 + B2) == pytest.approx(A + B, abs=1e-12)
    assert 0.0 <= float(moved) <= A


def test_ceff_energy_integral():
    # integral of ceff over T recovers c*range + Lfus (latent heat conserved)
    c, Lfus, Tm, dT = 1.0, 100.0, 0.0, 0.2
    T = jnp.linspace(-3.0, 3.0, 20001)
    integral = float(jnp.trapezoid(sm.ceff(T, c, Lfus, Tm, dT), T))
    expected = c * (3.0 - (-3.0)) + Lfus * float(sm.melt_fraction(jnp.float64(3.0), Tm, dT)
                                                 - sm.melt_fraction(jnp.float64(-3.0), Tm, dT))
    assert integral == pytest.approx(expected, rel=1e-4)

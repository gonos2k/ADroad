"""§7.4 Gauss-Newton (matrix-free, incremental 4D-Var) for state DA.

Recovers a multi-dimensional initial-temperature-profile perturbation from
surface-temperature observations (twin, free-running). GN uses only JVP/VJP
(no explicit Jacobian/Hessian) and converges in a few outer iterations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

jax = pytest.importorskip("jax")
from jax import config  # noqa: E402
config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from droad import jax_model as jm  # noqa: E402
from droad.assimilate import gauss_newton, fit  # noqa: E402
from tests.test_assimilate import _setup  # noqa: E402


def _apply(x0, d):
    # perturb initial layers 1..4 (near-surface, well observed)
    return (x0[0].at[1:5].add(d), x0[1].at[1:5].add(d), x0[2])


def test_gauss_newton_recovers_initial_profile():
    base, static, forc, x0, w = _setup(150)
    true_d = jnp.array([2.0, -1.5, 1.0, -0.5])
    obs = jm.dry_rollout(base, _apply(x0, true_d), forc, static)

    def residual(d):
        return jm.dry_rollout(base, _apply(x0, d), forc, static) - obs

    est = gauss_newton(residual, jnp.zeros(4), outer=4, cg_maxiter=40)
    assert np.allclose(np.array(est), np.array(true_d), atol=1e-4)


def test_gauss_newton_beats_first_order_in_few_iters():
    # GN reaches low loss in a handful of outer steps; report vs Adam budget.
    base, static, forc, x0, w = _setup(120)
    true_d = jnp.array([1.5, -1.0, 0.7, 0.3])
    obs = jm.dry_rollout(base, _apply(x0, true_d), forc, static)

    def residual(d):
        return jm.dry_rollout(base, _apply(x0, d), forc, static) - obs

    def loss(d):
        r = residual(d)
        return jnp.mean(r ** 2)

    gn = gauss_newton(residual, jnp.zeros(4), outer=3, cg_maxiter=40)
    loss_gn = float(loss(gn))

    adam, hist = fit(loss, jnp.zeros(4), steps=50, lr=0.1)   # small 1st-order budget
    loss_adam = float(loss(adam))

    assert loss_gn < 1e-8                     # GN essentially converged
    assert loss_gn < loss_adam                # and beats a short 1st-order run

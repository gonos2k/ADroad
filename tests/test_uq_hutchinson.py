"""§7.6 scalable UQ (M-H3): matrix-free Hutchinson Hessian-diagonal estimate.

Validates the stochastic estimator against the dense Hessian diagonal on a small
control (where dense is affordable), confirming the matrix-free path is correct.
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
from droad.assimilate import hutchinson_diag  # noqa: E402
from tests.test_assimilate import _setup, _offset_x0  # noqa: E402


def _loss_z():
    base, static, forc, x0, w = _setup(80)
    true = jnp.array([0.92, 1.5])
    obs = jm.dry_rollout({**base, "Emiss": true[0]}, _offset_x0(x0, true[1]), forc, static)

    def loss(z):
        return jm.loss({**base, "Emiss": z[0]}, _offset_x0(x0, z[1]), forc, obs, w, static)

    return loss


def test_hutchinson_matches_dense_hessian_diag():
    loss = _loss_z()
    z = jnp.array([0.88, 0.5])
    dense_diag = np.diag(np.array(jax.hessian(loss)(z)))
    est = np.array(hutchinson_diag(loss, z, n_samples=1200, seed=1))
    # unbiased but stochastic (variance from off-diagonal terms) -> realistic tol
    assert np.allclose(est, dense_diag, rtol=0.25, atol=0.06)


def test_posterior_std_positive():
    loss = _loss_z()
    z = jnp.array([0.90, 1.0])
    diag = np.array(hutchinson_diag(loss, z, n_samples=300, seed=2))
    assert np.all(diag > 0)                       # SPD near optimum
    std = 1.0 / np.sqrt(diag)                      # curvature-based std proxy
    assert np.all(np.isfinite(std))

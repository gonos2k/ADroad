"""2nd-order machinery (§7.4/7.6, §8): HVP, Newton, Laplace covariance.

Control vector z = [Emiss, initial-state offset]. Reuses the differentiable dry
model. HVP is forward-over-reverse; symmetry is the §8 correctness check.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402

jax = pytest.importorskip("jax")
from jax import config  # noqa: E402
config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from droad import jax_model as jm  # noqa: E402
from droad.assimilate import hvp, newton, laplace_cov  # noqa: E402
from tests.test_assimilate import _setup, _offset_x0  # noqa: E402


def _make_loss():
    base, static, forc, x0, w = _setup(80)
    true = jnp.array([0.92, 2.0])                     # [Emiss, offset]
    obs = jm.dry_rollout({**base, "Emiss": true[0]}, _offset_x0(x0, true[1]), forc, static)

    def loss(z):
        return jm.loss({**base, "Emiss": z[0]}, _offset_x0(x0, z[1]), forc, obs, w, static)

    return loss, true


def test_hvp_symmetry():
    loss, true = _make_loss()
    z = jnp.array([0.85, 0.0])
    u = jnp.array([1.0, -0.5])
    v = jnp.array([0.3, 0.7])
    Hv = hvp(loss, z, v)
    Hu = hvp(loss, z, u)
    assert float(jnp.dot(u, Hv)) == pytest.approx(float(jnp.dot(v, Hu)), rel=1e-8, abs=1e-10)


def test_hvp_matches_dense_hessian():
    loss, true = _make_loss()
    z = jnp.array([0.88, 1.0])
    v = jnp.array([1.0, 0.0])
    H = jax.hessian(loss)(z)
    assert np.allclose(np.array(hvp(loss, z, v)), np.array(H @ v), rtol=1e-8, atol=1e-10)


def test_newton_recovers_fast():
    loss, true = _make_loss()
    z_opt = newton(loss, jnp.array([0.85, 0.0]), steps=6)
    assert np.allclose(np.array(z_opt), np.array(true), atol=1e-3)
    assert float(loss(z_opt)) < 1e-10


def test_laplace_cov_spd_at_optimum():
    loss, true = _make_loss()
    z_opt = newton(loss, jnp.array([0.85, 0.0]), steps=8)
    cov = laplace_cov(loss, z_opt)
    cov = np.array(cov)
    assert np.allclose(cov, cov.T, atol=1e-8)          # symmetric
    assert np.all(np.linalg.eigvalsh(cov) > 0)         # positive definite
    assert np.all(np.sqrt(np.diag(cov)) > 0)           # finite std devs

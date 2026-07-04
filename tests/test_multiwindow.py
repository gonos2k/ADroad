"""§7.2 joint estimation: multi-window DA + shared global parameter.

W windows each carry their own initial-state control (offset); a single global
physical parameter (Emiss) is shared across all windows. One joint loss recovers
per-window states AND the global parameter simultaneously (the design's headline
"simultaneous DA + parameter optimization"). Windows run in parallel via vmap.
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
from droad.assimilate import fit  # noqa: E402
from tests.test_assimilate import _setup, _offset_x0  # noqa: E402


def test_joint_multiwindow_recovers_states_and_global_param():
    base, static, forc, x0, w = _setup(120)
    W = 3
    true_emiss = 0.93
    true_offs = jnp.array([2.0, -1.5, 0.8])         # per-window initial-state offsets

    def window_pred(emiss, off):
        return jm.dry_rollout({**base, "Emiss": emiss}, _offset_x0(x0, off), forc, static)

    # synthetic observations, one trajectory per window
    obs = jax.vmap(lambda off: window_pred(true_emiss, off))(true_offs)   # (W, T)

    def loss(control):
        preds = jax.vmap(lambda off: window_pred(control["Emiss"], off))(control["offs"])
        return jnp.mean((preds - obs) ** 2)

    init = {"Emiss": jnp.float64(0.85), "offs": jnp.zeros(W)}
    est, hist = fit(loss, init, steps=600, lr=0.02)

    assert float(est["Emiss"]) == pytest.approx(true_emiss, abs=2e-3)      # global param
    assert np.allclose(np.array(est["offs"]), np.array(true_offs), atol=3e-2)  # local states
    assert hist[-1] < hist[0] * 1e-4


def test_global_param_uses_all_windows():
    # gradient wrt the shared parameter aggregates contributions from every window
    base, static, forc, x0, w = _setup(80)
    W = 3
    offs = jnp.array([1.0, -1.0, 2.0])

    def window_pred(emiss, off):
        return jm.dry_rollout({**base, "Emiss": emiss}, _offset_x0(x0, off), forc, static)

    obs = jax.vmap(lambda o: window_pred(0.9, o))(offs)

    def loss_all(e):
        preds = jax.vmap(lambda o: window_pred(e, o))(offs)
        return jnp.mean((preds - obs) ** 2)

    def loss_one(e):   # only first window
        return jnp.mean((window_pred(e, offs[0]) - obs[0]) ** 2)

    g_all = float(jax.grad(loss_all)(jnp.float64(0.85)))
    g_one = float(jax.grad(loss_one)(jnp.float64(0.85)))
    assert np.isfinite(g_all) and abs(g_all) > 0
    # joint gradient differs from a single-window gradient (aggregates all)
    assert abs(g_all - g_one) > 1e-6

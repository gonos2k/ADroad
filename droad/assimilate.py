"""Gradient-based estimation driver (§7): variational DA + parameter calibration.

Minimizes a scalar loss over a pytree of control variables (physical params
and/or initial state) using optax. The gradient is a single reverse-mode pass
(VJP / adjoint), so cost is independent of the control dimension.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def hvp(f, x, v):
    """Hessian-vector product via forward-over-reverse (§7.4). Matrix-free."""
    return jax.jvp(jax.grad(f), (x,), (v,))[1]


def _sym(H):
    """Symmetrize a Hessian (kills asymmetry from finite-precision autodiff)."""
    return 0.5 * (H + H.T)


def newton(loss_fn, x0, steps=6, damping=1e-10):
    """Dense Newton for small control vectors (§7.4). Uses exact Hessian solve.
    The Hessian is symmetrized and Levenberg-damped before each solve."""
    x = x0
    eye = jnp.eye(x.shape[0])
    for _ in range(steps):
        g = jax.grad(loss_fn)(x)
        H = _sym(jax.hessian(loss_fn)(x))
        dz = jnp.linalg.solve(H + damping * eye, -g)
        x = x + dz
    return x


def laplace_cov(loss_fn, x_opt, damping=1e-10):
    """REGULARIZED Laplace posterior covariance ~ inv(Hessian) at optimum (§7.6).

    The Hessian is symmetrized, then shifted by enough to make it positive
    definite: shift = max(damping, damping - min_eigenvalue). At a true optimum
    the Hessian is already PSD and the shift is just `damping`; if it is
    indefinite (saddle / not converged) the eigenvalue floor still yields an SPD
    matrix to invert. Note this clips negative curvature — the result is a
    regularized covariance, not the raw inverse Hessian, so at a non-optimum it
    should be read as a stabilized approximation, not an exact posterior."""
    H = _sym(jax.hessian(loss_fn)(x_opt))
    eye = jnp.eye(H.shape[0])
    min_eval = jnp.min(jnp.linalg.eigvalsh(H))
    # shift = max(damping, damping - min_eval); via relu to avoid a raw jnp.maximum
    shift = damping + jax.nn.relu(-min_eval)
    return jnp.linalg.inv(H + shift * eye)


def hutchinson_diag(loss_fn, x, n_samples=200, seed=0):
    """Matrix-free stochastic estimate of diag(Hessian) via HVP (§7.6, M-H3).

    Unbiased Hutchinson estimator: E[v ⊙ Hv] with Rademacher v. Never forms H,
    so it scales to high-dimensional controls. Posterior variance ~ 1/diag.
    """
    key = jax.random.PRNGKey(seed)

    def one(k):
        v = jax.random.rademacher(k, x.shape, dtype=x.dtype)
        return v * hvp(loss_fn, x, v)

    return jnp.mean(jax.vmap(one)(jax.random.split(key, n_samples)), axis=0)


def gauss_newton(residual, z0, outer=5, cg_maxiter=40, damping=1e-8):
    """Matrix-free Gauss-Newton / incremental 4D-Var (§7.4).

    Minimizes 0.5*||residual(z)||^2. Each outer step linearizes once and solves
    (JᵀJ + damping) dz = -Jᵀ r by CG, using only JVP (Jv) and VJP (Jᵀv) — never
    forming J or the Hessian (scales to high-dim controls).
    """
    import jax.scipy.sparse.linalg as jssl

    z = z0
    for _ in range(outer):
        r0, Jv = jax.linearize(residual, z)      # Jv: tangent -> residual tangent
        JT = jax.linear_transpose(Jv, z)         # residual cotangent -> control

        def A(v):
            (jt,) = JT(Jv(v))
            return jt + damping * v

        (neg_b,) = JT(r0)
        dz, _ = jssl.cg(A, -neg_b, maxiter=cg_maxiter)
        z = z + dz
    return z


def fit(loss_fn, init, steps=300, lr=0.05, optimizer=None):
    """Minimize loss_fn(control) from `init` (any pytree).

    Returns (best_control, history): best_control is the iterate with the lowest
    observed loss (not simply the last), so a late divergent step can't undo a
    good solution. history is the list of scalar losses (loss at each iterate,
    measured before that step's update).

    Note: the final updated iterate is also evaluated as a best-candidate, but its
    loss is NOT appended to `history` — so if best_control is that final iterate,
    `min(history)` may be slightly above the returned iterate's loss. Use history
    for the optimization curve, not as the loss of best_control.
    """
    opt = optimizer if optimizer is not None else optax.adam(lr)
    control = init
    state = opt.init(control)
    value_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def update(control, state):
        loss, grad = value_and_grad(control)
        updates, state = opt.update(grad, state, control)
        return optax.apply_updates(control, updates), state, loss

    history = []
    best_control, best_loss = control, float("inf")
    for _ in range(steps):
        prev = control                       # iterate whose loss we measure
        control, state, loss = update(control, state)
        loss = float(loss)
        history.append(loss)
        if loss < best_loss:
            best_loss, best_control = loss, prev
    # also consider the final updated iterate (its loss was never measured in the
    # loop), so a last improving step isn't discarded — matters for small `steps`.
    final_loss = float(loss_fn(control))
    if final_loss < best_loss:
        best_control = control
    return best_control, history

"""dROAD — differentiable RoadSurf.

P0 scaffold. Backend-neutral (NumPy) first; JAX is activated only after
forward parity is locked (see 구현계획_P0_derisked.md §8).

API maturity: `config`, `branches`, `ledger` are the stabilized contracts. The
physics/DA modules (`storage`, `model`, `driver`, `jax_model`, `jax_storage`,
`assimilate`, `dual`) are exported for use but their signatures may still evolve.
"""

__all__ = [
    "config", "branches", "ledger",          # stabilized contracts
    "storage", "model", "driver",            # NumPy exact-mode physics
    "jax_model", "jax_storage",              # differentiable backend
    "assimilate", "dual",                    # DA / calibration
    "deviation",                             # deviation-budget aggregation
]

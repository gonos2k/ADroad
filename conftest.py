import sys
from pathlib import Path

# Make repo-root packages (droad, tools) importable when running pytest.
sys.path.insert(0, str(Path(__file__).parent))

# Auto-mark tests by module so `pytest -m "not jax"` runs the pure-NumPy core.
_JAX_MODULES = {
    "test_jax_dry", "test_jax_storage", "test_assimilate", "test_second_order",
    "test_multiwindow", "test_smoothing", "test_uq_hutchinson",
    "test_gauss_newton", "test_dual", "test_real_obs_da",
}
_REALDATA_MODULES = {"test_real_obs_da"}


def pytest_collection_modifyitems(config, items):
    import pytest
    for it in items:
        mod = it.module.__name__.rsplit(".", 1)[-1]
        if mod in _JAX_MODULES:
            it.add_marker(pytest.mark.jax)
        if mod in _REALDATA_MODULES:
            it.add_marker(pytest.mark.realdata)

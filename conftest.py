import sys
from pathlib import Path

# Make repo-root packages (droad, tools) importable when running pytest.
sys.path.insert(0, str(Path(__file__).parent))

# JAX/realdata tests declare `pytestmark = pytest.mark.jax` at module level
# (single source of truth), so `pytest -m "not jax"` runs only the NumPy core.

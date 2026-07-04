"""AST audit: ban raw branch/domain primitives in core (P0 §4).

Core code must route these through droad.branches wrappers instead of calling
`np.where`, `jnp.where`, `lax.cond`, `np.clip`, `np.sqrt`, `np.log`, `np.exp`,
`np.maximum`, `np.minimum` directly.

`droad/branches.py` is the ONE allowed place to call the raw primitives.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

BANNED_ATTRS = {
    "where", "clip", "sqrt", "log", "exp", "maximum", "minimum",  # np.*
    "cond",  # lax.cond
}
BANNED_MODULES = {"np", "numpy", "jnp", "lax"}

# files allowed to use raw primitives:
#  - branches.py : the sanctioned NumPy wrapper layer
#  - jax_model.py: the JAX backend (jnp primitives with inline domain guards)
DEFAULT_ALLOWLIST = {"branches.py", "jax_model.py", "smoothing.py", "jax_storage.py"}


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    call: str


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.found: list[Violation] = []

    def visit_Call(self, node: ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in BANNED_ATTRS:
            mod = f.value
            if isinstance(mod, ast.Name) and mod.id in BANNED_MODULES:
                self.found.append(
                    Violation(self.filename, node.lineno, f"{mod.id}.{f.attr}")
                )
        self.generic_visit(node)


def find_raw_primitives(package_dir, allowlist=DEFAULT_ALLOWLIST) -> list[Violation]:
    """Scan every .py under package_dir; return raw-primitive violations."""
    package_dir = Path(package_dir)
    violations: list[Violation] = []
    for path in sorted(package_dir.rglob("*.py")):
        if path.name in allowlist:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        v = _Visitor(path.name)
        v.visit(tree)
        violations.extend(v.found)
    return violations


def scan_source(source: str, filename: str = "<snippet>") -> list[Violation]:
    """Scan a source string (used by tests with a known-bad snippet)."""
    v = _Visitor(filename)
    v.visit(ast.parse(source, filename=filename))
    return v.found

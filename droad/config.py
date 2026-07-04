"""Run configuration: compatibility_target × model_mode (+ validation suites).

Two independent axes (P0 §1):
  - compatibility_target: what we must match at runtime  → {python, fortran}
  - model_mode:           which physics/approximation we run → {exact, smooth, enhanced}
  - validation_suite:     evaluation-only harnesses          → {paper, smooth_dev, forecast}

`paper_physics` is a validation suite, NOT a runtime target. Any model_mode
(including roadsurf_exact) may be evaluated in the paper suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompatibilityTarget(str, Enum):
    PYTHON = "python_compat"
    FORTRAN = "fortran_compat"


class ModelMode(str, Enum):
    EXACT = "roadsurf_exact"
    SMOOTH = "smooth_compat"
    ENHANCED = "enhanced_enthalpy_v1"


class ValidationSuite(str, Enum):
    PAPER = "paper_physics"
    SMOOTH_DEVIATION = "smooth_deviation"
    FORECAST_SKILL = "forecast_skill"


# Allowed runtime (target, mode) combinations. Everything else is rejected.
ALLOWED_RUNTIME: frozenset[tuple[CompatibilityTarget, ModelMode]] = frozenset({
    (CompatibilityTarget.PYTHON, ModelMode.EXACT),
    (CompatibilityTarget.FORTRAN, ModelMode.EXACT),
    (CompatibilityTarget.PYTHON, ModelMode.SMOOTH),
    (CompatibilityTarget.PYTHON, ModelMode.ENHANCED),
})


class ConfigError(ValueError):
    """Raised for any invalid run configuration."""


@dataclass(frozen=True)
class RunConfig:
    target: CompatibilityTarget
    mode: ModelMode
    validation_suites: tuple[ValidationSuite, ...] = ()


def _coerce(enum_cls, value, field: str):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(e.value for e in enum_cls)
        raise ConfigError(f"{field}={value!r} is not one of: {allowed}") from None


def validate_config(target, mode, validation_suites=()) -> RunConfig:
    """Validate and normalize a run configuration.

    Raises ConfigError if:
      - target is not a valid CompatibilityTarget (e.g. 'paper_physics' as target),
      - mode is not a valid ModelMode,
      - the (target, mode) pair is not allowed,
      - any validation suite name is unknown.
    """
    # 'paper_physics' as a runtime target is the classic mistake — give a clear message.
    if target in ("paper_physics", ValidationSuite.PAPER):
        raise ConfigError(
            "paper_physics is a validation_suite, not a runtime compatibility_target"
        )

    t = _coerce(CompatibilityTarget, target, "compatibility_target")
    m = _coerce(ModelMode, mode, "model_mode")
    suites = tuple(_coerce(ValidationSuite, s, "validation_suite") for s in validation_suites)

    if (t, m) not in ALLOWED_RUNTIME:
        raise ConfigError(f"combination not allowed: ({t.value}, {m.value})")

    return RunConfig(target=t, mode=m, validation_suites=suites)


def can_run_in_paper_suite(mode) -> bool:
    """Every model_mode (incl. roadsurf_exact) may run in the paper-physics suite."""
    _coerce(ModelMode, mode, "model_mode")
    return True

import pytest

from droad.config import (
    CompatibilityTarget,
    ModelMode,
    ValidationSuite,
    ConfigError,
    validate_config,
    can_run_in_paper_suite,
)


def test_valid_python_exact():
    cfg = validate_config("python_compat", "roadsurf_exact")
    assert cfg.target is CompatibilityTarget.PYTHON
    assert cfg.mode is ModelMode.EXACT


def test_valid_python_smooth_with_suites():
    cfg = validate_config(
        "python_compat", "smooth_compat", ["paper_physics", "forecast_skill"]
    )
    assert ValidationSuite.PAPER in cfg.validation_suites


def test_paper_physics_not_runtime_target():
    with pytest.raises(ConfigError):
        validate_config("paper_physics", "roadsurf_exact")


def test_roadsurf_exact_runs_in_paper_suite():
    assert can_run_in_paper_suite("roadsurf_exact") is True
    assert can_run_in_paper_suite(ModelMode.ENHANCED) is True


def test_forbidden_fortran_smooth():
    # fortran_compat only pairs with roadsurf_exact
    with pytest.raises(ConfigError):
        validate_config("fortran_compat", "smooth_compat")


def test_target_mode_combo_valid_rejects_unknown_mode():
    with pytest.raises(ConfigError):
        validate_config("python_compat", "enhanced_enthalpy_v2")

import numpy as np
import pytest

from droad import branches
from droad.branches import (
    BranchError,
    assert_branch_registered,
    safe_where,
    guarded_sqrt,
)
from tools.check_raw_primitives import find_raw_primitives, scan_source


# --- registry / wrapper contract ---

def test_registered_site_ok():
    out = safe_where("storage.min_clip", np.array([True, False]), 1.0, 0.0)
    assert list(out) == [1.0, 0.0]


def test_unregistered_site_fails():
    with pytest.raises(BranchError):
        safe_where("does.not.exist", True, 1.0, 0.0)


def test_wrong_policy_fails():
    # freeze_gate is 'custom_smooth', calling it as safe_where must fail
    with pytest.raises(BranchError):
        safe_where("storage.freeze_gate", True, 1.0, 0.0)


def test_guarded_sqrt_no_nan_on_negative():
    # inactive/negative input must not produce NaN
    out = guarded_sqrt("boundary_layer.psi_unstable", np.array([-5.0, 4.0]))
    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx(2.0)


def test_all_registry_policies_valid():
    for site, policy in branches.BRANCH_REGISTRY.items():
        assert_branch_registered(site, policy)  # should not raise


# --- raw primitive audit ---

def test_no_raw_primitives_in_core():
    import droad
    from pathlib import Path

    core = Path(droad.__file__).parent
    violations = find_raw_primitives(core)
    assert violations == [], f"raw primitive usage in core: {violations}"


def test_audit_detects_known_bad_example():
    bad = "import numpy as np\ndef f(x):\n    return np.where(x > 0, np.sqrt(x), 0.0)\n"
    found = scan_source(bad)
    calls = {v.call for v in found}
    assert "np.where" in calls and "np.sqrt" in calls

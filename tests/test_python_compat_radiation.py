"""G0a python_compat: droad.calc_rnet must match RoadSurf-Python bit-for-bit."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from droad.radiation import calc_rnet

RSP_SRC = Path(__file__).resolve().parent.parent / "reference" / "RoadSurf-Python" / "src"


@pytest.fixture(scope="module")
def ref_calc_rnet():
    sys.path.insert(0, str(RSP_SRC))
    import BalanceModel
    return BalanceModel.CalcRNet


CASES = [
    # emiss, sb, tsurf, albedo, sw, lw, cSW, cLW
    (0.95, 5.67e-8, -2.8, 0.1, 0.0, 250.0, 1.0, 1.0),
    (0.95, 5.67e-8, 5.0, 0.6, 400.0, 300.0, 1.0, 1.0),
    (0.95, 5.67e-8, 0.0, 0.1, 800.0, 350.0, 0.9, 1.1),
    (0.90, 5.67e-8, -15.0, 0.6, 50.0, 200.0, 1.0, 1.0),
]


@pytest.mark.parametrize("args", CASES)
def test_calc_rnet_matches_reference(ref_calc_rnet, args):
    emiss, sb, tsurf, albedo, sw, lw, csw, clw = args
    surf = SimpleNamespace(TsurfAve=tsurf)
    ref = ref_calc_rnet(emiss, sb, surf, albedo, sw, lw, csw, clw)
    got = calc_rnet(emiss, sb, tsurf, albedo, sw, lw, csw, clw)
    assert got == pytest.approx(ref, abs=1e-12, rel=0)

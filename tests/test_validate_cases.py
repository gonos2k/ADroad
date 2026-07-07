"""Independent-case manifest validator — pure schema/semantic contract (no jax)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.validate_cases import validate_case, validate_manifest, REGIMES  # noqa: E402


def _case(cid="s_2026-01-01_dry_cold", station="s", start="2026-01-01T00:00:00",
          end="2026-01-02T00:00:00", regime="dry_cold", **extra):
    return {"case_id": cid, "station": station, "start": start, "end": end,
            "regime": regime, **extra}


def test_valid_case_parses():
    start, end = validate_case(_case())
    assert start < end


def test_case_rejects_missing_fields_and_bad_regime():
    with pytest.raises(ValueError):
        validate_case({"case_id": "x", "station": "s"})            # missing start/end/regime
    with pytest.raises(ValueError):
        validate_case(_case(regime="blizzard"))                    # not in REGIMES
    with pytest.raises(ValueError):
        validate_case(_case(start="2026-01-02T00:00:00", end="2026-01-01T00:00:00"))  # start>=end
    with pytest.raises(ValueError):
        validate_case(_case(bogus="1"))                            # unknown field
    with pytest.raises(ValueError):
        validate_case(_case(start="not-a-date"))                   # bad ISO


def test_case_requires_time_and_typed_optionals():
    with pytest.raises(ValueError):
        validate_case(_case(start="2026-01-01"))                   # date-only, no time
    with pytest.raises(ValueError):
        validate_case(_case(forcing_source=123))                   # non-string optional
    with pytest.raises(ValueError):
        validate_case(_case(obs_source="  "))                      # empty non-empty-optional
    start, end = validate_case(_case(notes=""))                    # notes may be empty
    assert start < end


def test_two_tier_readiness():
    # 3 distinct non-overlapping cases across 3 regimes -> minimum yes, recommended no (<9)
    cases = [
        _case(cid="a1", station="a", start="2026-01-01T00:00:00",
              end="2026-01-02T00:00:00", regime="dry_cold"),
        _case(cid="b1", station="b", start="2026-01-05T00:00:00",
              end="2026-01-06T00:00:00", regime="precip_snow"),
        _case(cid="c1", station="c", start="2026-01-09T00:00:00",
              end="2026-01-10T00:00:00", regime="freeze_transition"),
    ]
    rep = validate_manifest({"cases": cases})
    assert rep["ok"] and rep["minimum_evidence_ready"] is True
    assert rep["recommended_promotion_ready"] is False             # 1 case/regime < 3
    assert any("per regime" in r for r in rep["readiness_reasons"])


def _cases_per_regime(counts):
    out, d = [], 1
    for regime, n in counts.items():
        for _ in range(n):
            out.append(_case(cid=f"{regime}_{d}", station=f"st{d}",
                             start=f"2026-03-{d:02d}T00:00:00", end=f"2026-03-{d:02d}T06:00:00",
                             regime=regime))
            d += 1
    return out


def test_recommended_requires_three_cases_per_regime():
    # balanced 3x3 -> recommended True
    ok = validate_manifest({"cases": _cases_per_regime(
        {"dry_cold": 3, "warm_wet": 3, "freeze_transition": 3})})
    assert ok["ok"] and ok["recommended_promotion_ready"] is True
    # lopsided 7/1/1 (9 total, 3 regimes) must FAIL recommended (per-regime, not bare 9)
    lop = validate_manifest({"cases": _cases_per_regime(
        {"dry_cold": 7, "warm_wet": 1, "freeze_transition": 1})})
    assert lop["ok"] and lop["minimum_evidence_ready"] is True
    assert lop["recommended_promotion_ready"] is False
    assert any("per regime" in r for r in lop["readiness_reasons"])


def test_minimum_not_met_when_too_few_or_single_regime():
    single_regime = [_case(cid=f"s{i}", start=f"2026-01-0{i}T00:00:00",
                           end=f"2026-01-0{i}T12:00:00") for i in (1, 2, 3)]
    rep = validate_manifest({"cases": single_regime})
    assert rep["ok"] is True and rep["minimum_evidence_ready"] is False   # one regime
    assert any("regimes" in r for r in rep["readiness_reasons"])
    rep2 = validate_manifest({"cases": single_regime[:1]})
    assert rep2["minimum_evidence_ready"] is False
    assert any("cases 1 < minimum" in r for r in rep2["readiness_reasons"])


def test_duplicate_id_is_error():
    rep = validate_manifest({"cases": [_case(), _case()]})         # same case_id twice
    assert rep["ok"] is False and any("duplicate case_id" in e for e in rep["errors"])


def test_same_station_overlap_is_not_independent():
    # distinct start dates but overlapping intervals at the same station -> error (not indep)
    overlap = [
        _case(cid="A1", station="A", start="2026-01-01T12:00:00", end="2026-01-02T12:00:00"),
        _case(cid="A2", station="A", start="2026-01-02T00:00:00", end="2026-01-03T00:00:00",
              regime="warm_wet"),
    ]
    rep = validate_manifest({"cases": overlap})
    assert rep["ok"] is False and any("overlapping cases" in e for e in rep["errors"])
    assert rep["minimum_evidence_ready"] is False


def test_mixed_timezone_does_not_raise():
    # tz-aware start vs naive end must be reported as an error, not blow up ("never raises")
    rep = validate_manifest({"cases": [
        _case(start="2026-01-01T00:00:00+00:00", end="2026-01-02T00:00:00")]})
    assert rep["ok"] is False and rep["minimum_evidence_ready"] is False


def test_manifest_rejects_non_mapping():
    rep = validate_manifest([])
    assert rep["ok"] is False and rep["minimum_evidence_ready"] is False


def test_example_yaml_is_schema_valid():
    yaml = pytest.importorskip("yaml")
    manifest = yaml.safe_load((REPO / "cases.example.yaml").read_text())
    rep = validate_manifest(manifest)
    assert rep["ok"] is True                                       # example is schema-clean
    assert set(rep["regimes"]).issubset(REGIMES)

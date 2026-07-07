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


def test_manifest_promotion_ready_needs_cases_regimes_independence():
    # 3 distinct station-days across 2 regimes -> ready as an evidence base
    cases = [
        _case(cid="a_2026-01-01_dry_cold", station="a", start="2026-01-01T00:00:00",
              end="2026-01-02T00:00:00", regime="dry_cold"),
        _case(cid="b_2026-01-05_precip_snow", station="b", start="2026-01-05T00:00:00",
              end="2026-01-06T00:00:00", regime="precip_snow"),
        _case(cid="c_2026-01-09_freeze_transition", station="c", start="2026-01-09T00:00:00",
              end="2026-01-10T00:00:00", regime="freeze_transition"),
    ]
    rep = validate_manifest({"cases": cases})
    assert rep["ok"] and rep["promotion_ready"] is True
    assert rep["n_cases"] == 3 and rep["n_regimes"] == 3 and rep["n_distinct_station_days"] == 3


def test_manifest_not_ready_when_too_few_or_single_regime():
    single_regime = [_case(cid=f"s_2026-01-0{i}_dry_cold", start=f"2026-01-0{i}T00:00:00",
                           end=f"2026-01-0{i+1}T00:00:00") for i in (1, 2, 3)]
    rep = validate_manifest({"cases": single_regime})
    assert rep["ok"] is True                                       # schema fine
    assert rep["promotion_ready"] is False                        # ...but only one regime
    assert any("regime coverage" in r for r in rep["promotion_reasons"])

    rep2 = validate_manifest({"cases": single_regime[:1]})
    assert rep2["promotion_ready"] is False
    assert any("insufficient cases" in r for r in rep2["promotion_reasons"])


def test_manifest_flags_duplicate_id_and_station_day():
    dup_id = [_case(), _case()]                                    # same case_id twice
    rep = validate_manifest({"cases": dup_id})
    assert rep["ok"] is False and any("duplicate case_id" in e for e in rep["errors"])

    # distinct ids but same station+date -> not independent -> not promotion_ready
    same_day = [_case(cid="x1"), _case(cid="x2"),
                _case(cid="y", station="t", start="2026-02-01T00:00:00",
                      end="2026-02-02T00:00:00", regime="warm_wet")]
    rep2 = validate_manifest({"cases": same_day})
    assert rep2["ok"] is True and rep2["promotion_ready"] is False
    assert any("station-day" in r for r in rep2["promotion_reasons"])


def test_manifest_rejects_non_mapping():
    rep = validate_manifest([])
    assert rep["ok"] is False and rep["promotion_ready"] is False


def test_example_yaml_is_schema_valid():
    yaml = pytest.importorskip("yaml")
    manifest = yaml.safe_load((REPO / "cases.example.yaml").read_text())
    rep = validate_manifest(manifest)
    assert rep["ok"] is True                                       # example is schema-clean
    assert set(rep["regimes"]).issubset(REGIMES)

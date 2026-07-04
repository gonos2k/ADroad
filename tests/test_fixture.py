"""Pinned no-coupling fixture regression (P0 §8.3 layer 1)."""

import hashlib
import json
from pathlib import Path

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "no_coupling"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_manifest_matches_output():
    manifest = json.loads((FIX / "manifest.json").read_text())
    assert _sha(FIX / "testi_output.csv") == manifest["output_sha256"]


def test_output_row_count():
    manifest = json.loads((FIX / "manifest.json").read_text())
    rows = sum(1 for _ in open(FIX / "testi_output.csv")) - 1
    assert rows == manifest["output_rows"] == 432

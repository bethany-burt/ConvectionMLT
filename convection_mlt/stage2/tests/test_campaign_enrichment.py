"""Campaign enrichment reconstructs metadata without mutating the raw JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PLOTS = Path(__file__).resolve().parents[1] / "plots"
if str(PLOTS) not in sys.path:
    sys.path.insert(0, str(PLOTS))

from campaign_spec import EXPECTED_ROLE_COUNTS, enrich_campaign_payload


def test_enrichment_classifies_case_27_as_pressure_range_check():
    raw_path = (
        Path(__file__).resolve().parents[1] / "results" / "production_campaign.json"
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    payload = enrich_campaign_payload(raw)
    assert payload["schema_version"] == 2
    assert payload["metadata_origin"] == "reconstructed_from_campaign_definition"
    assert "enthalpy_drift_tolerance" in payload["campaign_config"]
    cases = payload["cases"]
    assert len(cases) == 27
    roles = [c["campaign_role"] for c in cases]
    counts = {role: roles.count(role) for role in set(roles)}
    assert counts == EXPECTED_ROLE_COUNTS
    case27 = cases[26]
    assert case27["case_id"] == 27
    assert case27["campaign_role"] == "pressure_range_check"
    assert case27["pressure_bottom"] == 1.0e6
    assert case27["pressure_top"] == 1.0e2
    canonical = next(
        c
        for c in cases
        if c["campaign_role"] == "parameter_matrix"
        and c["n_layers"] == 50
        and c["x_he"] == 0.0
        and not c["irregular_grid"]
    )
    assert canonical["pressure_bottom"] == 1.0e7
    assert canonical["pressure_top"] == 1.0e3
    # Raw file is unchanged by enrichment (pure function).
    raw_again = json.loads(raw_path.read_text(encoding="utf-8"))
    assert "schema_version" not in raw_again
    assert "campaign_role" not in raw_again["cases"][0]

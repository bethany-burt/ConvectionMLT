"""Campaign specification used to enrich the raw production JSON.

The completed cluster artifact ``production_campaign.json`` is never modified.
Metadata here is reconstructed from the campaign definition, matching
``stage2/experiments/production_campaign.py``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

TEMPERATURE_CASE = "superadiabatic_nabla_0.30_Tbot_4000"
CANONICAL_P_BOTTOM = 1.0e7
CANONICAL_P_TOP = 1.0e3
SECONDARY_P_BOTTOM = 1.0e6
SECONDARY_P_TOP = 1.0e2
G0 = 10.0

CAMPAIGN_CONFIG = {
    "enthalpy_drift_tolerance": 1.0e-12,
    "isentrope_rms_tolerance": 1.0e-6,
    "epsilon_gradient": 1.0e-8,
    "theta_rms_tolerance": 1.0e-6,
    "flux_tolerance": 5.0e-3,
    "g0": G0,
    "temperature_case": TEMPERATURE_CASE,
}

ENRICHED_SCHEMA_VERSION = 2
METADATA_ORIGIN = "reconstructed_from_campaign_definition"
EXPECTED_TOTAL = 27
EXPECTED_ROLE_COUNTS = {
    "parameter_matrix": 24,
    "gravity_stress": 2,
    "pressure_range_check": 1,
}


def campaign_specs() -> list[dict[str, Any]]:
    """Return the 27 production specs in campaign order (case_id = index + 1)."""
    specs: list[dict[str, Any]] = []
    resolutions = (25, 50, 100, 200)
    x_he_values = (0.0, 0.10, 0.25)
    for n in resolutions:
        for x_he in x_he_values:
            for irregular in (False, True):
                specs.append(
                    {
                        "n_layers": n,
                        "x_he": x_he,
                        "gravity_mode": "constant",
                        "planet_radius": None,
                        "irregular_grid": irregular,
                        "pressure_bottom": CANONICAL_P_BOTTOM,
                        "pressure_top": CANONICAL_P_TOP,
                        "temperature_case": TEMPERATURE_CASE,
                        "campaign_role": "parameter_matrix",
                    }
                )
    for rp in (1.0e7, 1.0e8):
        specs.append(
            {
                "n_layers": 50,
                "x_he": 0.0,
                "gravity_mode": "inverse_square",
                "planet_radius": rp,
                "irregular_grid": False,
                "pressure_bottom": CANONICAL_P_BOTTOM,
                "pressure_top": CANONICAL_P_TOP,
                "temperature_case": TEMPERATURE_CASE,
                "campaign_role": "gravity_stress",
            }
        )
    specs.append(
        {
            "n_layers": 50,
            "x_he": 0.0,
            "gravity_mode": "constant",
            "planet_radius": None,
            "irregular_grid": False,
            "pressure_bottom": SECONDARY_P_BOTTOM,
            "pressure_top": SECONDARY_P_TOP,
            "temperature_case": TEMPERATURE_CASE,
            "campaign_role": "pressure_range_check",
        }
    )
    if len(specs) != EXPECTED_TOTAL:
        raise AssertionError(f"campaign definition has {len(specs)} cases, expected {EXPECTED_TOTAL}")
    return specs


def _values_match(raw: Any, spec: Any, *, key: str) -> bool:
    if raw is None and spec is None:
        return True
    if key in {"x_he", "planet_radius"} and raw is not None and spec is not None:
        return abs(float(raw) - float(spec)) <= 1.0e-12 * max(1.0, abs(float(spec)))
    return raw == spec


def enrich_campaign_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge reconstructed metadata onto a copy of the raw campaign payload."""
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError("raw campaign JSON is missing cases[]")
    specs = campaign_specs()
    if len(cases) != len(specs):
        raise ValueError(
            f"raw campaign has {len(cases)} cases; campaign definition has {len(specs)}"
        )

    enriched_cases = []
    for index, (case, spec) in enumerate(zip(cases, specs), start=1):
        for key in ("n_layers", "x_he", "gravity_mode", "irregular_grid"):
            raw_key = "irregular_grid" if key == "irregular_grid" else key
            if not _values_match(case.get(raw_key), spec[key], key=key):
                raise ValueError(
                    f"case {index}: raw {raw_key}={case.get(raw_key)!r} "
                    f"does not match campaign definition {spec[key]!r}"
                )
        if spec["gravity_mode"] == "inverse_square":
            if not _values_match(
                case.get("planet_radius"), spec["planet_radius"], key="planet_radius"
            ):
                raise ValueError(
                    f"case {index}: planet_radius {case.get('planet_radius')!r} "
                    f"does not match {spec['planet_radius']!r}"
                )
        merged = deepcopy(case)
        merged["case_id"] = index
        merged["pressure_bottom"] = spec["pressure_bottom"]
        merged["pressure_top"] = spec["pressure_top"]
        merged["temperature_case"] = spec["temperature_case"]
        merged["campaign_role"] = spec["campaign_role"]
        if "irregular_grid" not in merged:
            merged["irregular_grid"] = spec["irregular_grid"]
        enriched_cases.append(merged)

    payload = {
        "schema_version": ENRICHED_SCHEMA_VERSION,
        "metadata_origin": METADATA_ORIGIN,
        "campaign_config": dict(CAMPAIGN_CONFIG),
        "status": raw.get("status"),
        "completed": raw.get("completed"),
        "total": raw.get("total"),
        "max_steps": raw.get("max_steps"),
        "cases": enriched_cases,
        "failures": deepcopy(raw.get("failures", [])),
        "n_failures": raw.get("n_failures", len(raw.get("failures", []))),
        "raw_source": "production_campaign.json",
    }
    return payload

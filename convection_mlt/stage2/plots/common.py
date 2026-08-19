"""Shared helpers for Stage 2 validation data and Matplotlib figures."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from convection_mlt.metadata import dump_json, git_commit, git_dirty, json_safe

from campaign_spec import (
    ENRICHED_SCHEMA_VERSION,
    EXPECTED_ROLE_COUNTS,
    EXPECTED_TOTAL,
    METADATA_ORIGIN,
)

PLOTS_ROOT = Path(__file__).resolve().parent
DATA_DIR = PLOTS_ROOT / "data"
GENERATED_DIR = PLOTS_ROOT / "generated"
STAGE2_ROOT = PLOTS_ROOT.parent
RESULTS_DIR = STAGE2_ROOT / "results"
PACKAGE_ROOT = STAGE2_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parent
SRC_ROOT = PACKAGE_ROOT / "src"

RAW_CAMPAIGN_PATH = RESULTS_DIR / "production_campaign.json"
ENRICHED_CAMPAIGN_PATH = RESULTS_DIR / "production_campaign_enriched.json"

REQUIRED_CASE_FIELDS = (
    "case_id",
    "n_layers",
    "x_he",
    "gravity_mode",
    "irregular_grid",
    "pressure_bottom",
    "pressure_top",
    "temperature_case",
    "campaign_role",
    "status",
    "steps",
    "enthalpy_drift",
    "temperature_rms_vs_isentrope",
    "max_superadiabaticity",
    "max_z_over_rp",
)
REQUIRED_CONFIG_FIELDS = (
    "enthalpy_drift_tolerance",
    "isentrope_rms_tolerance",
    "epsilon_gradient",
)


class MissingSourceError(FileNotFoundError):
    """A required saved data file is absent or incomplete."""


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    dump_json(path, payload)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise MissingSourceError(f"missing required source file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_source(path: Path, *, description: str) -> Path:
    if not path.exists():
        raise MissingSourceError(f"missing {description}: {path}")
    if path.stat().st_size <= 0:
        raise MissingSourceError(f"incomplete {description}: {path} is empty")
    return path


def require_finite(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def campaign_config(payload: dict[str, Any]) -> dict[str, float]:
    config = payload.get("campaign_config")
    if not isinstance(config, dict):
        raise ValueError("enriched campaign JSON is missing campaign_config")
    missing = [key for key in REQUIRED_CONFIG_FIELDS if key not in config]
    if missing:
        raise ValueError(f"campaign_config missing fields: {missing}")
    return {
        key: require_finite(f"campaign_config.{key}", config[key])
        for key in REQUIRED_CONFIG_FIELDS
    }


def validate_enriched_campaign(payload: dict[str, Any]) -> dict[str, Any]:
    """Hard-fail preflight for make_all. Returns the payload if valid."""
    schema = payload.get("schema_version")
    if schema != ENRICHED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {schema!r}; expected {ENRICHED_SCHEMA_VERSION}"
        )
    origin = payload.get("metadata_origin")
    if origin != METADATA_ORIGIN:
        raise ValueError(
            f"unexpected metadata_origin {origin!r}; expected {METADATA_ORIGIN!r}"
        )
    campaign_config(payload)
    for key in ("status", "completed", "total", "cases"):
        if key not in payload:
            raise ValueError(f"enriched campaign JSON missing {key}")
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("enriched campaign JSON has empty cases[]")
    completed = int(payload["completed"])
    total = int(payload["total"])
    if not (completed == total == EXPECTED_TOTAL == len(cases)):
        raise ValueError(
            f"campaign completeness failed: completed={completed} total={total} "
            f"n_cases={len(cases)} expected={EXPECTED_TOTAL}"
        )
    ids: list[int] = []
    roles: list[str] = []
    for index, case in enumerate(cases, start=1):
        missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
        if missing:
            raise ValueError(f"case {index} missing fields: {missing}")
        case_id = int(case["case_id"])
        ids.append(case_id)
        roles.append(str(case["campaign_role"]))
        require_finite(f"case {case_id}.enthalpy_drift", case["enthalpy_drift"])
        require_finite(
            f"case {case_id}.temperature_rms_vs_isentrope",
            case["temperature_rms_vs_isentrope"],
        )
        require_finite(
            f"case {case_id}.max_superadiabaticity", case["max_superadiabaticity"]
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate case_id values: {ids}")
    if sorted(ids) != list(range(1, EXPECTED_TOTAL + 1)):
        raise ValueError(f"case_id values are not 1..{EXPECTED_TOTAL}: {ids}")
    counts = Counter(roles)
    if dict(counts) != EXPECTED_ROLE_COUNTS:
        raise ValueError(
            f"unexpected campaign_role counts {dict(counts)}; "
            f"expected {EXPECTED_ROLE_COUNTS}"
        )
    return payload


def load_enriched_campaign() -> dict[str, Any]:
    require_source(ENRICHED_CAMPAIGN_PATH, description="enriched campaign JSON")
    payload = read_json(ENRICHED_CAMPAIGN_PATH)
    return validate_enriched_campaign(payload)


def parameter_matrix_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for case in payload["cases"] if case["campaign_role"] == "parameter_matrix"]


def cases_with_role(payload: dict[str, Any], role: str) -> list[dict[str, Any]]:
    return [case for case in payload["cases"] if case["campaign_role"] == role]


def format_significant(value: float, *, digits: int = 4) -> str:
    number = require_finite("significant_value", value)
    if 0.0 < number < 1.0:
        text = f"{number:.{digits}g}"
        if float(text) >= 1.0:
            return f"<1.{'0' * max(digits - 1, 3)}"
        return text
    return f"{number:.{digits}g}"


def format_score_ratio(value: float, *, decimals: int = 4) -> str:
    number = require_finite("score_ratio", value)
    if number < 1.0:
        text = f"{number:.{decimals}f}"
        if float(text) >= 1.0:
            return f"<1.{'0' * decimals}"
        return text
    return f"{number:.{decimals}f}"


def fit_log_log_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n_points": 0}
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_points": int(np.count_nonzero(mask)),
    }


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.constrained_layout.use": True,
        }
    )


def pressure_axis(ax) -> None:
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("P [Pa]")


def code_revision() -> dict[str, Any]:
    return {
        "git_commit": git_commit(REPO_ROOT),
        "git_dirty": git_dirty(REPO_ROOT),
    }


def write_figure_metadata(
    stem: str,
    *,
    source_files: list[Path | str],
    tolerances: dict[str, float] | None,
    cases_included: list[int] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    ensure_dirs()
    payload = {
        "figure": stem,
        "source_files": [str(path) for path in source_files],
        "tolerances": tolerances or {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision(),
        "cases_included": cases_included,
    }
    if extra:
        payload["extra"] = json_safe(extra)
    path = GENERATED_DIR / f"{stem}.meta.json"
    write_json(path, payload)
    return path


def save_figure(fig, stem: str, *, bbox_inches: str | None = None, **metadata: Any) -> list[Path]:
    """Write PNG plus a metadata sidecar."""
    ensure_dirs()
    path = GENERATED_DIR / f"{stem}.png"
    save_kwargs: dict[str, Any] = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
        save_kwargs["pad_inches"] = 0.2
    fig.savefig(path, **save_kwargs)
    write_figure_metadata(stem, **metadata)
    return [path]

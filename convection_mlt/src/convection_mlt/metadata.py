"""Deterministic run metadata and strict JSON serialisation."""

from dataclasses import asdict, is_dataclass
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def git_commit(repository: Path | None = None) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty(repository: Path | None = None) -> bool | None:
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(config: Any, repository: Path | None = None) -> dict[str, Any]:
    payload = asdict(config) if is_dataclass(config) else dict(config)
    physics = payload.get("physics", payload)
    prefactor = physics.get("closure_prefactor", 0.5)
    return {
        "git_commit": git_commit(repository),
        "git_dirty": git_dirty(repository),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "precision": "float64",
        "flux_sign": "upward_positive",
        "pressure_order": "bottom_to_top_decreasing",
        "closure": {
            "name": "AGNI_Lee_inspired_R0",
            "mixing_length": "alpha_times_pressure_scale_height",
            "prefactor": prefactor,
            "sources": [
                {
                    "name": "AGNI atmospheric convection documentation",
                    "url": (
                        "https://www.h-nicholls.space/AGNI/dev/"
                        "explanation/model_convection/"
                    ),
                    "accessed": "2026-08-09",
                    "formulae": ["convective_flux", "convective_velocity"],
                },
                {
                    "name": "Lee, Tan & Tsai (2024)",
                    "doi": "10.1093/mnras/stae537",
                    "formulae": ["pressure_coordinate_MLT", "Kzz=w*ell"],
                },
            ],
        },
        "units": {
            "pressure": "Pa",
            "temperature": "K",
            "density": "kg m^-3",
            "heat_capacity": "J kg^-1 K^-1",
            "length": "m",
            "velocity": "m s^-1",
            "flux": "W m^-2",
            "diffusivity": "m^2 s^-1",
            "time": "s",
        },
        "configuration": payload,
    }


def json_safe(value: Any) -> Any:
    """Convert arrays/scalars recursively; nonfinite values become JSON null."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

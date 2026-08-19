"""Render Stage 2 validation figures from saved data only.

Does not rerun diagnostic calculations. Fails if required sources are missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "convection-mlt-matplotlib-stage2"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PLOTS_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PLOTS_ROOT.parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
STAGE2_ROOT = PLOTS_ROOT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PLOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTS_ROOT))
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))

from common import (  # noqa: E402
    DATA_DIR,
    ENRICHED_CAMPAIGN_PATH,
    GENERATED_DIR,
    MissingSourceError,
    ensure_dirs,
    load_enriched_campaign,
    require_source,
)

PLOT_SCRIPTS = (
    "plot_thermo_audit.py",
    "plot_representative_isentrope.py",
    "plot_robustness_matrix.py",
    "plot_resolution_robustness.py",
    "plot_enthalpy_conservation.py",
    "plot_hydrostatic_verification.py",
    "plot_inverse_square.py",
    "plot_step_scaling.py",
    "plot_audit_table.py",
)

REQUIRED_DATA = {
    "plot_thermo_audit.py": [DATA_DIR / "thermo_audit.json"],
    "plot_representative_isentrope.py": [DATA_DIR / "representative_column.json"],
    "plot_robustness_matrix.py": [ENRICHED_CAMPAIGN_PATH],
    "plot_resolution_robustness.py": [ENRICHED_CAMPAIGN_PATH],
    "plot_enthalpy_conservation.py": [ENRICHED_CAMPAIGN_PATH],
    "plot_hydrostatic_verification.py": [DATA_DIR / "hydro_references.json"],
    "plot_inverse_square.py": [ENRICHED_CAMPAIGN_PATH, DATA_DIR / "gravity_limit.json"],
    "plot_step_scaling.py": [ENRICHED_CAMPAIGN_PATH],
    "plot_audit_table.py": [DATA_DIR / "audit.json"],
}

REQUIRED_OUTPUTS = (
    "fig01_thermo_audit.png",
    "fig02_representative_isentrope.png",
    "fig03_robustness_matrix.png",
    "fig04_resolution_robustness.png",
    "fig05_enthalpy_conservation.png",
    "fig06_hydrostatic_verification.png",
    "fig07_inverse_square.png",
    "fig08_step_scaling.png",
    "fig09_audit_table.png",
)


def _plot_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(SRC_ROOT), str(PLOTS_ROOT), str(STAGE2_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]
    return env


def _preflight() -> None:
    load_enriched_campaign()
    for script, paths in REQUIRED_DATA.items():
        for path in paths:
            require_source(path, description=f"{script} source")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Stage 2 figures from saved data")
    parser.parse_args()
    ensure_dirs()
    try:
        _preflight()
    except MissingSourceError as exc:
        raise SystemExit(f"make_all preflight failed: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"make_all preflight failed: {exc}") from exc

    env = _plot_env()
    for script in PLOT_SCRIPTS:
        path = PLOTS_ROOT / script
        print(f"==> {script}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PLOTS_ROOT),
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"{script} failed with exit code {completed.returncode}")

    missing = [name for name in REQUIRED_OUTPUTS if not (GENERATED_DIR / name).exists()]
    if missing:
        raise SystemExit(f"make_all finished but missing outputs: {missing}")
    print(f"wrote figures in {GENERATED_DIR}")


if __name__ == "__main__":
    main()

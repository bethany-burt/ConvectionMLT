"""Write synthetic analytic κ(P) HELIOS premixed HDF5 for nested columns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import nested_analytic_opacity_spec
from convection_mlt.adapters.helios_contracts import MICROBAR_TO_BAR
from convection_mlt.adapters.helios_grid import build_helios_pressure_grid
from convection_mlt.adapters.helios_opacity_table import (
    build_table_arrays,
    table_checksum,
    write_helios_opacity_hdf5,
)

FIXTURES = ROOT / "fixtures" / "helios"
DEFAULT_OUT = FIXTURES / "analytic_grey_nested.h5"


def main(layers: tuple[int, ...] = (96, 192), out: Path = DEFAULT_OUT) -> dict:
    t_min = 1.0e30
    t_max = 0.0
    p_min_bar = 1.0e30
    p_max_bar = 0.0
    opacity = None
    for n in layers:
        spec = nested_analytic_opacity_spec(n)
        rec_path = ROOT / "results" / (
            "n192_implicit_rce.json" if n == 192 else "nested_rce_family.json"
        )
        if n == 96:
            members = json.loads(rec_path.read_text()).get("members") or {}
            rec = members["96"]
        else:
            rec = json.loads(rec_path.read_text())
        t = np.asarray(rec["temperature"], dtype=np.float64)
        p = np.asarray(rec["pressure_centres"], dtype=np.float64)
        t_min = min(t_min, float(np.min(t)))
        t_max = max(t_max, float(np.max(t)))
        p_min_bar = min(p_min_bar, float(np.min(p)) / 1.0e5)
        p_max_bar = max(p_max_bar, float(np.max(p)) / 1.0e5)
        opacity = spec.opacity()
        ref = ROOT / "fixtures" / "helios" / f"helios_grid_n{n}_thermal_reference.json"
        if ref.exists():
            g = json.loads(ref.read_text())["grid"]
            grid = build_helios_pressure_grid(
                p_boa_microbar=float(g["p_boa_microbar"]),
                p_toa_microbar=float(g["p_toa_microbar"]),
                n_layers=n,
            )
            p_min_bar = min(p_min_bar, float(np.min(grid.p_lay_microbar)) * MICROBAR_TO_BAR)
            p_max_bar = max(p_max_bar, float(np.max(grid.p_lay_microbar)) * MICROBAR_TO_BAR)
    assert opacity is not None
    table = build_table_arrays(
        opacity,
        t_min=t_min * 0.9,
        t_max=t_max * 1.1,
        p_min_bar=p_min_bar * 0.5,
        p_max_bar=p_max_bar * 2.0,
        n_press=256,
    )
    checksum = write_helios_opacity_hdf5(out, table)
    meta = {
        "out": str(out),
        "checksum_sha256": checksum,
        "schema_version": table.schema_version,
        "kpoints_flatten": "helios",
        "layers_used_for_hull": list(layers),
        "temperature_range_K": [float(table.temperatures_k[0]), float(table.temperatures_k[-1])],
        "pressure_range_bar": [float(table.pressures_bar[0]), float(table.pressures_bar[-1])],
        "metadata": table.metadata,
    }
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layers", type=int, nargs="+", default=[96, 192])
    args = parser.parse_args()
    main(tuple(args.layers), args.out)

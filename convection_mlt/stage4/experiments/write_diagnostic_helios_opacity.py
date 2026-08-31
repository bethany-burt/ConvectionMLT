"""Write constant-κ and pressure-tagged HELIOS HDF5 tables (same flatten as analytic)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import nested_analytic_opacity_spec
from convection_mlt.adapters.helios_contracts import MICROBAR_TO_BAR
from convection_mlt.adapters.helios_grid import build_helios_pressure_grid
from convection_mlt.adapters.helios_opacity_table import (
    build_constant_opacity_table,
    build_pressure_tagged_table,
    write_helios_opacity_hdf5,
)
from export_helios_grid_reference import _load_record

FIXTURES = ROOT / "fixtures" / "helios"
CONSTANT_OUT = FIXTURES / "constant_grey.h5"
TAGGED_OUT = FIXTURES / "pressure_tagged.h5"


def _hull(layers: tuple[int, ...] = (96, 192)) -> tuple[float, float, float, float]:
    t_min, t_max = 1.0e30, 0.0
    p_min_bar, p_max_bar = 1.0e30, 0.0
    for n in layers:
        rec = _load_record(n)
        t = np.asarray(rec["temperature"], dtype=np.float64)
        p = np.asarray(rec["pressure_centres"], dtype=np.float64)
        t_min = min(t_min, float(np.min(t)))
        t_max = max(t_max, float(np.max(t)))
        p_min_bar = min(p_min_bar, float(np.min(p)) / 1.0e5)
        p_max_bar = max(p_max_bar, float(np.max(p)) / 1.0e5)
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
    return t_min * 0.9, t_max * 1.1, p_min_bar * 0.5, p_max_bar * 2.0


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constant-out", type=Path, default=CONSTANT_OUT)
    parser.add_argument("--tagged-out", type=Path, default=TAGGED_OUT)
    parser.add_argument("--constant-kappa-si", type=float, default=None)
    args = parser.parse_args()

    t_min, t_max, p_min_bar, p_max_bar = _hull()
    spec = nested_analytic_opacity_spec(96)
    kappa_si = float(args.constant_kappa_si if args.constant_kappa_si is not None else spec.opacity().kappa0)

    constant = build_constant_opacity_table(
        kappa_si, t_min=t_min, t_max=t_max, p_min_bar=p_min_bar, p_max_bar=p_max_bar,
        n_press=256,
    )
    tagged = build_pressure_tagged_table(
        t_min=t_min, t_max=t_max, p_min_bar=p_min_bar, p_max_bar=p_max_bar,
        n_press=256,
    )
    csum = write_helios_opacity_hdf5(args.constant_out, constant)
    tsum = write_helios_opacity_hdf5(args.tagged_out, tagged)
    payload = {
        "constant": {
            "out": str(args.constant_out),
            "checksum_sha256": csum,
            "kappa_si": kappa_si,
            "flatten": "helios",
            "metadata": constant.metadata,
        },
        "pressure_tagged": {
            "out": str(args.tagged_out),
            "checksum_sha256": tsum,
            "flatten": "helios",
            "metadata": tagged.metadata,
        },
        "note": (
            "kpoints use HELIOS host order (y fastest). Constant-κ isolates "
            "transport/BC; pressure-tagged exposes axis/order mistakes."
        ),
    }
    args.constant_out.with_suffix(".json").write_text(json.dumps(payload["constant"], indent=2) + "\n")
    args.tagged_out.with_suffix(".json").write_text(json.dumps(payload["pressure_tagged"], indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()

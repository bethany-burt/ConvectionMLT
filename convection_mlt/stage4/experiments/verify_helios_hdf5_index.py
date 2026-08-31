"""Offline HDF5 round-trip against HELIOS host linear-index formulas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt.adapters.helios_opacity_table import (
    build_unique_axis_table,
    helios_kpoints_flat_index,
    helios_meanmolmass_flat_index,
    helios_rayleigh_flat_index,
    read_helios_opacity_hdf5,
    unique_kpoints_encoding,
    write_helios_opacity_hdf5,
)

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("h5py is required") from exc


def verify_helios_index(path: Path) -> dict:
    table = read_helios_opacity_hdf5(path)
    ny, nx, npress, ntemp = table.kpoints_cgs.shape
    with h5py.File(path, "r") as f:
        flat = np.asarray(f["kpoints"][:], dtype=np.float64)
        mean_flat = np.asarray(f["meanmolmass"][:], dtype=np.float64)
        ray_flat = np.asarray(f["weighted Rayleigh cross-sections"][:], dtype=np.float64)
        formula = str(f.attrs.get("linear_index_formula", ""))
    k_ok = True
    mu_ok = True
    ray_ok = True
    for t in range(ntemp):
        for p in range(npress):
            i_mu = helios_meanmolmass_flat_index(p, t, npress=npress)
            mu_ok = mu_ok and mean_flat[i_mu] == table.mean_mol_mass_kg[p, t]
            for x in range(nx):
                i_ray = helios_rayleigh_flat_index(x, p, t, nx=nx, npress=npress)
                ray_ok = ray_ok and ray_flat[i_ray] == table.rayleigh_cross[x, p, t]
                for y in range(ny):
                    i = helios_kpoints_flat_index(y, x, p, t, ny=ny, nx=nx, npress=npress)
                    k_ok = k_ok and flat[i] == table.kpoints_cgs[y, x, p, t]
    status = "PASS" if k_ok and mu_ok and ray_ok else "FAIL"
    return {
        "status": status,
        "path": str(path),
        "kpoints_index_ok": bool(k_ok),
        "meanmolmass_index_ok": bool(mu_ok),
        "rayleigh_index_ok": bool(ray_ok),
        "linear_index_formula": formula,
        "shape": [int(ny), int(nx), int(npress), int(ntemp)],
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Write a unique-axis table here.")
    parser.add_argument("--table", type=Path, default=None, help="Existing HDF5 to audit.")
    args = parser.parse_args()
    if args.table is not None:
        payload = verify_helios_index(args.table)
    else:
        table = build_unique_axis_table()
        out = args.out or Path("/tmp/helios_unique_axis.h5")
        write_helios_opacity_hdf5(out, table)
        payload = verify_helios_index(out)
        ny, nx, npress, ntemp = table.kpoints_cgs.shape
        payload["encoding"] = "y + 10*x + 1000*p + 100000*t"
        payload["sample"] = unique_kpoints_encoding(1, 2, 3, 1)
        payload["shape"] = [ny, nx, npress, ntemp]
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("HELIOS HDF5 index round-trip failed")
    return payload


if __name__ == "__main__":
    main()

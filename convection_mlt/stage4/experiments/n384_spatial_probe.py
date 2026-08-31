"""192→384 spatial diagnostics from stored N=384 snapshots. Not a formal PASS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import ConstantH2Thermo
from convection_mlt.energy import column_enthalpy_per_area

from run_nested_family import interpolate_temperature

RESULTS = ROOT / "results"
N192 = RESULTS / "n192_implicit_rce.json"
OUT = RESULTS / "n384_spatial_probe.json"
GATES = {"max_rel_T": 0.02, "rcb_dex": 0.05}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _seed_from_n192(n192: dict, n384: dict) -> dict:
    t = interpolate_temperature(
        np.log(np.asarray(n192["pressure_centres"], dtype=np.float64)),
        n192["temperature"],
        np.log(np.asarray(n384["pressure_centres"], dtype=np.float64)),
    )
    h = ConstantH2Thermo().enthalpy(t)
    return {
        "n_layers": 384,
        "status": "interpolated_seed",
        "steps_accepted": 0,
        "flux_flatness": None,
        "tendency_norm": None,
        "primary_rcb_log10p": n192.get("primary_rcb_log10p"),
        "pressure_centres": n384["pressure_centres"],
        "temperature": t.tolist(),
        "mass_path": n384["mass_path"],
        "enthalpy": np.asarray(h, dtype=np.float64).tolist(),
        "column_enthalpy": column_enthalpy_per_area(n384["mass_path"], h),
    }


def _topology(record: dict) -> dict:
    regions = record.get("convective_regions") or []
    detached = record.get("detached_convective_regions") or []
    n = int(record.get("n_layers") or 0)
    bottom_connected = bool(regions) and int(regions[0][0]) == 0
    return {
        "n_convective_regions": len(regions),
        "convective_regions": regions,
        "detached_convective_regions": detached,
        "n_detached": len(detached),
        "bottom_connected": bottom_connected,
        "top_convective_layer": None if not regions else int(regions[-1][1]),
        "convective_fraction_of_layers": (
            None if not regions or n <= 0 else (int(regions[-1][1]) + 1) / n
        ),
    }


def _compare(coarse: dict, fine: dict) -> dict:
    p_c = np.asarray(coarse["pressure_centres"], dtype=np.float64)
    p_f = np.asarray(fine["pressure_centres"], dtype=np.float64)
    t_c = np.asarray(coarse["temperature"], dtype=np.float64)
    t_f_on_c = interpolate_temperature(np.log(p_f), fine["temperature"], np.log(p_c))
    scale = np.maximum(np.abs(t_c), 1.0)
    rel = np.abs(t_f_on_c - t_c) / scale
    i = int(np.argmax(rel))
    rcb_c = coarse.get("primary_rcb_log10p")
    rcb_f = fine.get("primary_rcb_log10p")
    h_c = float(
        coarse.get("column_enthalpy")
        or column_enthalpy_per_area(coarse["mass_path"], coarse["enthalpy"])
    )
    h_f = float(
        fine.get("column_enthalpy")
        or column_enthalpy_per_area(fine["mass_path"], fine["enthalpy"])
    )
    d_rcb = None if rcb_c is None or rcb_f is None else float(rcb_f) - float(rcb_c)
    return {
        "coarse_n": coarse.get("n_layers"),
        "fine_n": fine.get("n_layers"),
        "fine_status": fine.get("status"),
        "fine_steps_accepted": fine.get("steps_accepted"),
        "fine_flux_flatness": fine.get("flux_flatness"),
        "fine_tendency_norm": fine.get("tendency_norm"),
        "max_rel_T_on_coarse_P": float(rel[i]),
        "max_rel_T_layer_index": i,
        "max_rel_T_pressure_Pa": float(p_c[i]),
        "max_rel_T_log10p": float(np.log10(p_c[i])),
        "max_rel_T_T_coarse_K": float(t_c[i]),
        "max_rel_T_T_fine_on_coarse_K": float(t_f_on_c[i]),
        "median_rel_T_on_coarse_P": float(np.median(rel)),
        "coarse_rcb_log10p": rcb_c,
        "fine_rcb_log10p": rcb_f,
        "delta_log10_P_rcb": d_rcb,
        "abs_delta_log10_P_rcb": None if d_rcb is None else abs(d_rcb),
        "column_enthalpy_coarse": h_c,
        "column_enthalpy_fine": h_f,
        "column_enthalpy_rel": abs(h_f - h_c) / max(abs(h_c), abs(h_f), 1.0),
        "coarse_topology": _topology(coarse),
        "fine_topology": _topology(fine),
        "same_single_bottom_connected_zone": (
            _topology(coarse)["n_convective_regions"] == 1
            and _topology(fine)["n_convective_regions"] == 1
            and _topology(coarse)["n_detached"] == 0
            and _topology(fine)["n_detached"] == 0
            and _topology(coarse)["bottom_connected"]
            and _topology(fine)["bottom_connected"]
        ),
        "within_T_gate": float(rel[i]) <= GATES["max_rel_T"],
        "within_rcb_gate": d_rcb is not None and abs(d_rcb) <= GATES["rcb_dex"],
        "formal_spatial_pass": False,
        "note": (
            "Informal 192→384 comparison. Formal spatial PASS still requires "
            "physically gated N=384 (flatness and tendency ≤ 1e-3)."
        ),
    }


def main() -> dict:
    n192 = _load(N192)
    latest = _load(RESULTS / "n384_implicit_rce.json")
    snapshots = {
        "step0_interpolated_seed": _seed_from_n192(n192, latest),
    }
    for path in sorted(RESULTS.glob("n384_probe_step*.json")):
        rec = _load(path)
        snapshots[f"step{int(rec['steps_accepted']):05d}"] = rec
    snapshots[f"step{int(latest['steps_accepted']):05d}_latest"] = latest

    comparisons = {name: _compare(n192, rec) for name, rec in snapshots.items()}
    ordered = sorted(
        ((name, row) for name, row in comparisons.items() if row["fine_steps_accepted"] is not None),
        key=lambda item: int(item[1]["fine_steps_accepted"]),
    )
    evolution = []
    for name, row in ordered:
        evolution.append({
            "snapshot": name,
            "steps_accepted": row["fine_steps_accepted"],
            "flux_flatness": row["fine_flux_flatness"],
            "max_rel_T_on_coarse_P": row["max_rel_T_on_coarse_P"],
            "abs_delta_log10_P_rcb": row["abs_delta_log10_P_rcb"],
            "column_enthalpy_rel": row["column_enthalpy_rel"],
            "max_rel_T_log10p": row["max_rel_T_log10p"],
            "n_convective_regions": (row.get("fine_topology") or {}).get("n_convective_regions"),
            "n_detached": (row.get("fine_topology") or {}).get("n_detached"),
        })
    latest_row = comparisons[f"step{int(latest['steps_accepted']):05d}_latest"]
    payload = {
        "gates": GATES,
        "n192": {
            "status": n192.get("status"),
            "steps_accepted": n192.get("steps_accepted"),
            "flux_flatness": n192.get("flux_flatness"),
            "primary_rcb_log10p": n192.get("primary_rcb_log10p"),
            "profile_checksum_sha256": n192.get("profile_checksum_sha256"),
        },
        "n384_latest": {
            "status": latest.get("status"),
            "steps_accepted": latest.get("steps_accepted"),
            "flux_flatness": latest.get("flux_flatness"),
            "tendency_norm": latest.get("tendency_norm"),
            "primary_rcb_log10p": latest.get("primary_rcb_log10p"),
            "last_accepted_dt": latest.get("last_accepted_dt"),
            "energy_gate_ratio": latest.get("energy_gate_ratio"),
            "profile_checksum_sha256": latest.get("profile_checksum_sha256"),
        },
        "comparisons": comparisons,
        "evolution": evolution,
        "latest_comparison": latest_row,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "out": str(OUT),
        "n384_steps": latest.get("steps_accepted"),
        "max_rel_T": latest_row["max_rel_T_on_coarse_P"],
        "max_rel_T_log10p": latest_row["max_rel_T_log10p"],
        "abs_drcb": latest_row["abs_delta_log10_P_rcb"],
        "enthalpy_rel": latest_row["column_enthalpy_rel"],
        "within_T_gate": latest_row["within_T_gate"],
        "within_rcb_gate": latest_row["within_rcb_gate"],
        "flatness": latest.get("flux_flatness"),
        "evolution": evolution,
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()

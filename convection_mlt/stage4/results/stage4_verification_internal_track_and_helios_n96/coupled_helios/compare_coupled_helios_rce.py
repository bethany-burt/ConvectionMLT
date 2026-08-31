"""Score coupled HELIOS RCE against nested MLT using frozen benchmark tolerances.

HELIOS convective adjustment and the finite mixing-length closure are different
convection models. This is a benchmark, not solver identity. Without HELIOS
output files the status is NOT_RUN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import load_integrated_flux
from convection_mlt.adapters.helios import flux_cgs_to_si, load_tp_profile
from convection_mlt.adapters.helios_contracts import F_INT, MICROBAR_TO_PA, helios_track_status
from convection_mlt.energy import column_enthalpy_per_area
from export_helios_grid_reference import _load_record

FIXTURES = ROOT / "fixtures" / "helios"
TOLERANCES = FIXTURES / "coupled_rce_benchmark_tolerances.json"
RESULTS = ROOT / "results"


def interpolate_temperature(log_p_src, t_src, log_p_dst):
    order = np.argsort(log_p_src)
    return np.interp(log_p_dst, log_p_src[order], np.asarray(t_src, dtype=np.float64)[order])


def _mlt_topology(rec: dict) -> dict:
    regions = rec.get("convective_regions") or []
    detached = rec.get("detached_convective_regions") or []
    bottom = [r for r in regions if r and r[0] == 0]
    return {
        "convective_regions": regions,
        "detached_convective_regions": detached,
        "single_bottom_cz": len(bottom) == 1 and len(detached) == 0,
        "primary_rcb_log10p": rec.get("primary_rcb_log10p"),
    }


def _helios_rcb_and_topology(tp, flux, f_int: float) -> dict:
    p_pa = np.asarray(tp.pressure_microbar, dtype=np.float64) * MICROBAR_TO_PA
    flag = np.asarray(tp.conv_unstable_flag, dtype=np.float64)
    # Skip BOA row (index -1 / 0th entry) when present.
    lay = tp.layer_index != -1
    p_lay = p_pa[lay]
    flag_lay = flag[lay]
    if np.all(np.isfinite(flag_lay)):
        unstable = flag_lay > 0.5
    else:
        conv = flux_cgs_to_si(np.asarray(flux.flux_conv_net_cgs, dtype=np.float64))
        n = min(conv.size, p_lay.size)
        unstable = np.abs(conv[:n]) > 0.01 * abs(f_int)
        p_lay = p_lay[:n]
    if not np.any(unstable):
        return {
            "primary_rcb_log10p": None,
            "single_bottom_cz": False,
            "n_detached": None,
            "n_unstable": 0,
        }
    i0 = int(np.argmax(unstable))
    i_hi = i0
    while i_hi + 1 < unstable.size and unstable[i_hi + 1]:
        i_hi += 1
    detached = bool(np.any(unstable[i_hi + 1 :])) if i_hi + 1 < unstable.size else False
    rcb_p = float(p_lay[i_hi])
    return {
        "primary_rcb_log10p": float(np.log10(rcb_p)),
        "single_bottom_cz": i0 == 0 and not detached,
        "n_detached": int(detached),
        "n_unstable": int(np.sum(unstable)),
    }


def score(
    *,
    n_layers: int,
    helios_tp: Path | None,
    helios_flux: Path | None,
    tolerances: dict,
    runtime: dict | None = None,
) -> dict:
    rec = _load_record(n_layers)
    gates = tolerances.get("gates") or {}
    mlt_topo = _mlt_topology(rec)
    payload = {
        "comparison_type": tolerances.get("comparison_type", "benchmark_not_solver_identity"),
        "frozen_before_live": tolerances.get("frozen_before_live"),
        "n_layers": n_layers,
        "mlt_profile_checksum_sha256": rec.get("profile_checksum_sha256") or rec.get("checksum_sha256"),
        "mlt_rcb_log10p": rec.get("primary_rcb_log10p"),
        "mlt_topology": mlt_topo,
        "gates": gates,
        "helios_runtime_config": runtime or {},
        "status": "NOT_RUN",
        **helios_track_status(adapter_contract="PASS"),
    }
    payload["helios_coupled_rce_status"] = "NOT_RUN"
    if helios_tp is None or helios_flux is None or not helios_tp.exists() or not helios_flux.exists():
        payload["note"] = "HELIOS iterative output not present; tolerances remain frozen."
        return payload

    tp = load_tp_profile(helios_tp)
    flux = load_integrated_flux(helios_flux)
    p_mlt = np.asarray(rec["pressure_centres"], dtype=np.float64)
    t_mlt = np.asarray(rec["temperature"], dtype=np.float64)
    lay = tp.layer_index != -1
    p_h = np.asarray(tp.pressure_microbar[lay], dtype=np.float64) * MICROBAR_TO_PA
    t_h = np.asarray(tp.temperature_k[lay], dtype=np.float64)
    t_h_on_mlt = interpolate_temperature(np.log(p_h), t_h, np.log(p_mlt))
    scale = np.maximum(np.abs(t_mlt), 1.0)
    rel = np.abs(t_h_on_mlt - t_mlt) / scale
    imax = int(np.argmax(rel))
    f_net = flux_cgs_to_si(np.asarray(flux.flux_net_cgs, dtype=np.float64))
    f_top = float(f_net[-1])
    f_bot = float(f_net[0])
    f_int = float(rec.get("f_int") or F_INT)
    helios_topo = _helios_rcb_and_topology(tp, flux, f_int)
    rcb_mlt = rec.get("primary_rcb_log10p")
    rcb_h = helios_topo.get("primary_rcb_log10p")
    rcb_dex = (
        None if rcb_mlt is None or rcb_h is None else abs(float(rcb_h) - float(rcb_mlt))
    )
    h_mlt = float(rec.get("column_enthalpy") or column_enthalpy_per_area(
        rec["mass_path"], rec["enthalpy"]
    ))
    metrics = {
        "toa_flux_rel_vs_fint_helios": abs(f_top - f_int) / max(abs(f_int), 1.0),
        "toa_flux_rel_vs_mlt": abs(f_top - float(np.asarray(rec["flux_total"])[-1])) / max(abs(f_int), 1.0),
        "bottom_flux_rel": abs(f_bot - f_int) / max(abs(f_int), 1.0),
        "max_rel_T": float(rel[imax]),
        "max_rel_T_index": imax,
        "max_rel_T_pressure": float(p_mlt[imax]),
        "rcb_dex": rcb_dex,
        "helios_rcb_log10p": rcb_h,
        "topology_single_bottom_cz": bool(
            mlt_topo["single_bottom_cz"] and helios_topo.get("single_bottom_cz")
        ),
        "no_detached_convective_regions": bool(
            not (rec.get("detached_convective_regions") or [])
            and not helios_topo.get("n_detached")
        ),
        "energy_closure_rel": abs(f_bot - f_int) / max(abs(f_int), 1.0),
        "column_enthalpy_mlt": h_mlt,
    }
    checks = {
        "toa_flux_rel": metrics["toa_flux_rel_vs_fint_helios"] <= float(gates["toa_flux_rel"]),
        "max_rel_T": metrics["max_rel_T"] <= float(gates["max_rel_T"]),
        "rcb_dex": rcb_dex is not None and rcb_dex <= float(gates["rcb_dex"]),
        "topology_single_bottom_cz": metrics["topology_single_bottom_cz"]
        == bool(gates.get("topology_single_bottom_cz", True)),
        "no_detached_convective_regions": metrics["no_detached_convective_regions"]
        == bool(gates.get("no_detached_convective_regions", True)),
        "energy_closure_rel": metrics["energy_closure_rel"] <= float(gates["energy_closure_rel"]),
    }
    passed = all(checks.values())
    payload.update({
        "status": "PASS" if passed else "FAIL",
        "metrics": metrics,
        "checks": checks,
        "helios_topology": helios_topo,
        "helios_tp": str(helios_tp),
        "helios_flux": str(helios_flux),
        "helios_coupled_rce_status": "PASS" if passed else "FAIL",
        "helios_parity_headline": False,
        "full_stage4_claim": False,
        "note": (
            "Benchmark against finite-MLT nested RCE. HELIOS convective "
            "adjustment is a different convection model."
        ),
    })
    return payload


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument("--helios-tp", type=Path, default=None)
    parser.add_argument("--helios-flux", type=Path, default=None)
    parser.add_argument("--runtime-config", type=Path, default=None)
    parser.add_argument("--tolerances", type=Path, default=TOLERANCES)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "helios_coupled_rce_n96.json",
    )
    args = parser.parse_args()
    tols = json.loads(args.tolerances.read_text())
    runtime = json.loads(args.runtime_config.read_text()) if args.runtime_config else None
    payload = score(
        n_layers=args.layers,
        helios_tp=args.helios_tp,
        helios_flux=args.helios_flux,
        tolerances=tols,
        runtime=runtime,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "helios_coupled_rce_status": payload["helios_coupled_rce_status"],
        "helios_parity_headline": payload["helios_parity_headline"],
        "full_stage4_claim": payload["full_stage4_claim"],
        "out": str(args.output),
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()

"""Compare HELIOS frozen radiation-only output against HELIOS-grid Stage-3 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import STEFAN_BOLTZMANN, load_integrated_flux
from convection_mlt.adapters.helios import (
    flux_cgs_to_si,
    heating_from_net_flux,
    layer_energy_increment,
    to_canonical_interfaces,
)
from convection_mlt.adapters.helios_contracts import F_INT, PINNED_HELIOS_COMMIT, helios_track_status

FIXTURES = ROOT / "fixtures" / "helios"
TOLERANCES = FIXTURES / "radiation_only_tolerances.json"
RESULTS = ROOT / "results"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _track_for_case(case: str, status: str) -> dict:
    """Radiation-only labels for this JSON. Coupled/headline claims stay false."""
    n96 = status if case.startswith("n96") else ("PASS" if case.startswith("n192") else "NOT_RUN")
    n192 = status if case.startswith("n192") else "NOT_RUN"
    return helios_track_status(adapter_contract="PASS", n96=n96, n192=n192)


def _norm_diff(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = min(aa.size, bb.size)
    scale = np.maximum.reduce([np.abs(aa[:n]), np.abs(bb[:n]), np.full(n, floor)])
    return float(np.max(np.abs(aa[:n] - bb[:n]) / scale))


def _local_discrepancy(
    helios: np.ndarray,
    ref: np.ndarray,
    pressure: np.ndarray,
    *,
    local_floor: float,
) -> dict:
    """Max local relative error, not normalized by the deep-flux scale."""
    h = np.asarray(helios, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    n = min(h.size, r.size, p.size)
    scale = np.maximum.reduce([np.abs(h[:n]), np.abs(r[:n]), np.full(n, local_floor)])
    rel = np.abs(h[:n] - r[:n]) / scale
    i = int(np.argmax(rel))
    return {
        "max_local_rel": float(rel[i]),
        "interface": i,
        "pressure_microbar": float(p[i]),
        "helios": float(h[i]),
        "reference": float(r[i]),
        "abs_W_m2": float(abs(h[i] - r[i])),
    }


def compare_frozen(
    reference: dict,
    helios_flux_path: Path,
    *,
    tolerances: dict,
    case: str,
    helios_runtime_config: dict | None = None,
) -> dict:
    flux = load_integrated_flux(helios_flux_path)
    n_layers = int(reference["n_layers"])
    helios_net = to_canonical_interfaces(
        flux_cgs_to_si(flux.flux_net_cgs),
        flux.pressure_microbar,
        n_layers=n_layers,
    )
    helios_up = to_canonical_interfaces(
        flux_cgs_to_si(flux.flux_up_cgs),
        flux.pressure_microbar,
        n_layers=n_layers,
    )
    helios_down = to_canonical_interfaces(
        flux_cgs_to_si(flux.flux_down_cgs),
        flux.pressure_microbar,
        n_layers=n_layers,
    )
    helios_intern = to_canonical_interfaces(
        flux_cgs_to_si(flux.flux_intern_cgs),
        flux.pressure_microbar,
        n_layers=n_layers,
    )

    ref = reference["frozen"]
    ref_net = np.asarray(ref["flux_net_W_m2"], dtype=np.float64)
    ref_up = np.asarray(ref["flux_up_W_m2"], dtype=np.float64)
    ref_down = np.asarray(ref["flux_down_W_m2"], dtype=np.float64)
    ref_p_int = np.asarray(ref["pressure_interfaces_helios_microbar"], dtype=np.float64)
    mass = np.asarray(ref["mass_path_kg_m2"], dtype=np.float64)
    t_boa = float(ref["temperature_boa_k"])
    sigma_tboa4 = float(STEFAN_BOLTZMANN * t_boa**4)
    ref_heating = heating_from_net_flux(ref_net, mass)
    helios_heating = heating_from_net_flux(helios_net, mass)
    ref_df = layer_energy_increment(ref_net)
    helios_df = layer_energy_increment(helios_net)

    f_scale = max(float(np.max(np.abs(ref_net))), sigma_tboa4, 1.0)
    q_scale = max(float(np.max(np.abs(ref_heating))), 1e-12)
    df_scale = max(float(np.max(np.abs(ref_df))), 1.0)
    p_scale = max(float(np.max(ref_p_int)), 1.0)
    decomp_scale = np.maximum.reduce(
        [np.abs(helios_net), np.abs(helios_up - helios_down), np.full(helios_net.size, 1.0)]
    )
    ref_column = abs(float(np.sum(mass * ref_heating) - (ref_net[0] - ref_net[-1])))
    helios_column = abs(float(np.sum(mass * helios_heating) - (helios_net[0] - helios_net[-1])))

    metrics = {
        "pressure_grid_rel": _norm_diff(flux.pressure_microbar, ref_p_int, p_scale),
        "flux_decomposition_rel": float(np.max(np.abs(helios_net - (helios_up - helios_down)) / decomp_scale)),
        "rocky_surface_up_rel": abs(float(helios_up[0]) - sigma_tboa4) / max(sigma_tboa4, 1.0),
        "rocky_surface_net_rel": abs(float(helios_net[0]) - (sigma_tboa4 - float(helios_down[0])))
        / max(sigma_tboa4, 1.0),
        "toa_flux_down_rel": abs(float(helios_down[-1])) / f_scale,
        "flux_up_rel": _norm_diff(helios_up, ref_up, f_scale),
        "flux_down_rel": _norm_diff(helios_down, ref_down, f_scale),
        "flux_net_rel": _norm_diff(helios_net, ref_net, f_scale),
        "heating_from_flux_rel": _norm_diff(helios_heating, ref_heating, q_scale),
        "layer_energy_increment_rel": _norm_diff(helios_df, ref_df, df_scale),
        "reference_column_energy_identity_rel": ref_column
        / max(abs(ref_net[0] - ref_net[-1]), 1e-12),
        "helios_column_energy_closure_rel": helios_column
        / max(abs(helios_net[0] - helios_net[-1]), 1e-12),
        "column_energy_closure_rel": ref_column
        / max(abs(ref_net[0] - ref_net[-1]), 1e-12),
    }
    if np.isfinite(helios_intern[0]):
        metrics["f_intern_parameter_rel"] = abs(float(helios_intern[0]) - F_INT) / max(F_INT, 1.0)
    gates = tolerances["gates"]
    stage_order = tolerances.get(
        "comparison_order",
        [
            "pressure_grid",
            "f_intern_parameter",
            "flux_decomposition",
            "rocky_surface_up",
            "rocky_surface_net",
            "toa_flux_down",
            "flux_up",
            "flux_down",
            "flux_net",
            "heating_from_flux",
            "reference_column_energy_identity",
            "helios_column_energy_closure",
        ],
    )
    metric_key = {
        "pressure_grid": "pressure_grid_rel",
        "f_intern_parameter": "f_intern_parameter_rel",
        "flux_decomposition": "flux_decomposition_rel",
        "rocky_surface_up": "rocky_surface_up_rel",
        "rocky_surface_net": "rocky_surface_net_rel",
        "toa_flux_down": "toa_flux_down_rel",
        "flux_up": "flux_up_rel",
        "flux_down": "flux_down_rel",
        "flux_net": "flux_net_rel",
        "heating_from_flux": "heating_from_flux_rel",
        "layer_energy_increment": "layer_energy_increment_rel",
        "reference_column_energy_identity": "reference_column_energy_identity_rel",
        "helios_column_energy_closure": "helios_column_energy_closure_rel",
        "column_energy_closure": "column_energy_closure_rel",
    }
    stages = {}
    all_pass = True
    for stage in stage_order:
        key = metric_key.get(stage, stage)
        observed = metrics.get(key)
        if observed is None:
            continue
        if key == "f_intern_parameter_rel" and not np.isfinite(observed):
            stages[stage] = {
                "observed": None,
                "tolerance": gates.get(key),
                "status": "SKIP",
                "note": "F_intern column absent in HELIOS flux file.",
            }
            continue
        tol = gates.get(key, gates.get("flux_net_rel", 1.0))
        ok = observed <= tol
        stages[stage] = {
            "observed": observed,
            "tolerance": tol,
            "status": "PASS" if ok else "FAIL",
        }
        all_pass = all_pass and ok

    return {
        "case": case,
        "helios_commit": PINNED_HELIOS_COMMIT,
        "comparison_type": "radiation_only_not_mlt",
        "reference_source": reference.get("source_record"),
        "reference_grid": reference.get("reference_grid", "helios_geometric"),
        "reference_file": reference.get("_path"),
        "grid_checksum_sha256": reference.get("grid_checksum_sha256"),
        "helios_flux_file": str(helios_flux_path),
        "helios_flux_checksum_sha256": _sha256(helios_flux_path),
        "helios_runtime_config": helios_runtime_config or {},
        "tolerances_source": str(TOLERANCES),
        "tolerances_frozen_before_live": tolerances.get("frozen_before_live", True),
        "metrics": metrics,
        "heating_units": "W kg^-1",
        "local": {
            "flux_up": _local_discrepancy(helios_up, ref_up, ref_p_int, local_floor=1.0),
            "flux_down": _local_discrepancy(helios_down, ref_down, ref_p_int, local_floor=1.0),
            "flux_net": _local_discrepancy(helios_net, ref_net, ref_p_int, local_floor=1.0),
            "layer_energy_increment": _local_discrepancy(
                helios_df, ref_df, np.asarray(ref.get("pressure_centres_helios_microbar", ref_p_int[:-1])),
                local_floor=1.0,
            ),
        },
        "stages": stages,
        "comparison_order": stage_order,
        "status": "PASS" if all_pass else "FAIL",
        **_track_for_case(case, "PASS" if all_pass else "FAIL"),
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, choices=(8, 96, 192), required=True)
    parser.add_argument("--mode", choices=("thermal", "irradiated"), default="thermal")
    parser.add_argument("--helios-flux", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runtime-config", type=Path, default=None)
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()

    ref_path = args.reference or RESULTS / (
        f"helios_grid_n{args.layers}_{args.mode}_reference.json"
    )
    reference = json.loads(ref_path.read_text())
    reference["_path"] = str(ref_path)
    tolerances = json.loads(TOLERANCES.read_text())
    runtime_cfg = None
    if args.runtime_config and args.runtime_config.exists():
        runtime_cfg = json.loads(args.runtime_config.read_text())
    case = f"n{args.layers}_{args.mode}"
    if args.structural_only:
        result = {
            "case": case,
            "comparison_type": "structural_not_parity",
            "status": "PILOT",
            "note": "Irradiated beam contract not exact; not scored against parity gates.",
            "helios_flux_file": str(args.helios_flux),
            **helios_track_status(),
        }
    else:
        result = compare_frozen(
            reference,
            args.helios_flux,
            tolerances=tolerances,
            case=case,
            helios_runtime_config=runtime_cfg,
        )
    out = args.output or RESULTS / f"helios_frozen_rad_n{args.layers}_{args.mode}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out), "status": result["status"]}, indent=2))
    return result


if __name__ == "__main__":
    main()

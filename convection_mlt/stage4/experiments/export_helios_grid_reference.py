"""Export Stage-3 radiation reference on the HELIOS geometric grid."""

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

from convection_mlt import (
    ConstantGreyOpacity,
    LowerTemperature,
    STEFAN_BOLTZMANN,
    TopIrradiation,
    nested_analytic_opacity_spec,
    solve_radiation,
)
from convection_mlt.adapters.helios import (
    flux_si_to_cgs,
    heating_from_net_flux,
    write_tp_profile_from_grid,
)
from convection_mlt.adapters.helios_contracts import (
    F_INT,
    GRAVITY_SI,
    HELIOS_DEFAULT_DIFFUSIVITY,
    PINNED_HELIOS_COMMIT,
    PROVENANCE_ONLY,
    STAGE3_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    T_INT,
    TP_PRESSURE_UNIT,
)
from convection_mlt.adapters.helios_grid import (
    build_helios_grid_from_nested_edges,
    sample_nested_tp_on_helios_grid,
    top_interface_from_toa_center,
)

RESULTS = ROOT / "results"
NESTED = RESULTS / "nested_rce_family.json"
N192 = RESULTS / "n192_implicit_rce.json"


def _load_record(n_layers: int) -> dict:
    if n_layers == 192:
        return json.loads(N192.read_text())
    members = json.loads(NESTED.read_text()).get("members") or {}
    key = str(n_layers)
    if key not in members:
        raise FileNotFoundError(f"nested member N={n_layers} not in {NESTED}")
    return members[key]


def _grid_checksum(grid) -> str:
    payload = {
        "n_layers": grid.n_layers,
        "p_boa": grid.p_boa_microbar,
        "p_toa": grid.p_toa_microbar,
        "p_lay": grid.p_lay_microbar.tolist(),
        "p_int": grid.p_int_microbar.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def export_helios_grid_reference(
    n_layers: int,
    *,
    thermal_only: bool,
    diffusivity: float,
    opacity_mode: str = "analytic",
    constant_kappa_si: float | None = None,
) -> dict:
    rec = _load_record(n_layers)
    spec = nested_analytic_opacity_spec(n_layers)
    edges = np.asarray(rec["pressure_edges"], dtype=np.float64)
    grid = build_helios_grid_from_nested_edges(edges, n_layers)
    temperature_boa_k, temperature_lay_k = sample_nested_tp_on_helios_grid(rec, grid)
    f_irr = 0.0 if thermal_only else spec.f_irr
    mode = "thermal_only" if thermal_only else "irradiated"
    if opacity_mode == "constant":
        kappa0 = float(constant_kappa_si if constant_kappa_si is not None else spec.opacity().kappa0)
        opacity = ConstantGreyOpacity(kappa0)
        opacity_label = f"ConstantGreyOpacity(kappa0={kappa0})"
    elif opacity_mode == "analytic":
        opacity = spec.opacity()
        opacity_label = "nested_analytic_opacity_spec"
    else:
        raise ValueError(f"unknown opacity_mode: {opacity_mode}")
    rad = solve_radiation(
        temperature_lay_k,
        grid.layer_mass_kg_m2,
        opacity,
        grid.p_lay_Pa,
        TopIrradiation(f_irr),
        LowerTemperature(float(temperature_boa_k)),
        diffusivity_factor=diffusivity,
    )
    f_net = np.asarray(rad.flux_net, dtype=np.float64)
    f_up = np.asarray(rad.flux_up.sum(axis=0), dtype=np.float64)
    f_down = np.asarray(rad.flux_down.sum(axis=0), dtype=np.float64)
    heating = heating_from_net_flux(f_net, grid.layer_mass_kg_m2)
    payload = {
        "purpose": "Stage-3 radiation reference on HELIOS geometric grid for radiation-only parity",
        "coupled_helios_rce_claimed": False,
        "comparison_type": "radiation_only_not_mlt",
        "reference_grid": "helios_geometric",
        "mode": mode,
        "n_layers": n_layers,
        "source_record": "n192_implicit_rce.json" if n_layers == 192 else f"nested_rce_family.json[{n_layers}]",
        "source_status": rec.get("status"),
        "profile_checksum_sha256": rec.get("profile_checksum_sha256"),
        "grid_checksum_sha256": _grid_checksum(grid),
        "helios_commit": PINNED_HELIOS_COMMIT,
        "contracts": {
            "gravity_si": GRAVITY_SI,
            "f_int": F_INT,
            "f_irr": f_irr,
            "t_int_K": T_INT,
            "internal_flux_temperature_k": T_INT,
            "diffusivity_factor": diffusivity,
            "opacity": opacity_label,
            "opacity_mode": opacity_mode,
            "lower_bc": "LowerTemperature(T_boa)",
            "helios_lower_bc": "rocky_black_surface_zero_albedo",
            "helios_internal_flux_parameter_W_m2": F_INT,
            "bottom_convective_flux": 0.0,
            "canonical_orientation": "bottom_first_same_as_helios",
            "helios_orientation": "bottom_first_same_as_canonical",
            "tp_pressure_unit": TP_PRESSURE_UNIT,
            "flux_unit_canonical": "W m^-2",
            "flux_unit_helios": "erg s^-1 cm^-2",
            "provenance_only": PROVENANCE_ONLY,
        },
        "grid": {
            "p_boa_microbar": grid.p_boa_microbar,
            "p_toa_microbar": grid.p_toa_microbar,
            "p_top_int_microbar": float(top_interface_from_toa_center(
                grid.p_toa_microbar, grid.p_boa_microbar, grid.n_layers
            )),
            "p_lay_microbar": grid.p_lay_microbar.tolist(),
            "p_int_microbar": grid.p_int_microbar.tolist(),
            "p_lay_Pa": grid.p_lay_Pa.tolist(),
            "p_int_Pa": grid.p_int_Pa.tolist(),
            "layer_mass_kg_m2": grid.layer_mass_kg_m2.tolist(),
        },
        "frozen": {
            "temperature_boa_k": temperature_boa_k,
            "temperature_lay_k": temperature_lay_k.tolist(),
            "mass_path_kg_m2": grid.layer_mass_kg_m2.tolist(),
            "pressure_interfaces_helios_microbar": grid.p_int_microbar.tolist(),
            "pressure_centres_helios_microbar": grid.p_lay_microbar.tolist(),
            "flux_up_W_m2": f_up.tolist(),
            "flux_down_W_m2": f_down.tolist(),
            "flux_net_W_m2": f_net.tolist(),
            "heating_W_kg": heating.tolist(),
            "heating_W_m2": heating.tolist(),
            "flux_net_helios_erg_s_cm2": flux_si_to_cgs(f_net).tolist(),
        },
        "diagnostics": {
            "flux_net_bottom": float(f_net[0]),
            "flux_up_bottom": float(f_up[0]),
            "flux_down_bottom": float(f_down[0]),
            "flux_net_top": float(f_net[-1]),
            "sigma_Tboa4": float(STEFAN_BOLTZMANN * temperature_boa_k**4),
            "rocky_surface_up_residual_rel": float(
                abs(f_up[0] - STEFAN_BOLTZMANN * temperature_boa_k**4)
                / max(STEFAN_BOLTZMANN * temperature_boa_k**4, 1.0)
            ),
            "rocky_surface_net_residual_rel": float(
                abs(f_net[0] - (STEFAN_BOLTZMANN * temperature_boa_k**4 - f_down[0]))
                / max(STEFAN_BOLTZMANN * temperature_boa_k**4, 1.0)
            ),
            "temperature_boa_minus_t_int_K": float(temperature_boa_k - T_INT),
        },
        "sensitivity": {
            "stage3_diffusivity_D166": STAGE3_DIFFUSIVITY,
            "helios_default_diffusivity": HELIOS_DEFAULT_DIFFUSIVITY,
        },
    }
    return payload


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, choices=(8, 96, 192), required=True)
    parser.add_argument(
        "--mode",
        choices=("thermal-only", "irradiated"),
        default="thermal-only",
    )
    parser.add_argument("--diffusivity", type=float, default=HELIOS_DEFAULT_DIFFUSIVITY)
    parser.add_argument("--write-tp", type=Path, default=None)
    parser.add_argument("--opacity-mode", choices=("analytic", "constant"), default="analytic")
    parser.add_argument("--constant-kappa-si", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    thermal = args.mode == "thermal-only"
    payload = export_helios_grid_reference(
        args.layers,
        thermal_only=thermal,
        diffusivity=args.diffusivity,
        opacity_mode=args.opacity_mode,
        constant_kappa_si=args.constant_kappa_si,
    )
    suffix = f"n{args.layers}_{'thermal' if thermal else 'irradiated'}"
    if args.opacity_mode == "constant":
        suffix = f"{suffix}_constant"
    out = args.output or RESULTS / f"helios_grid_{suffix}_reference.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.write_tp is not None:
        from convection_mlt.adapters.helios_grid import build_helios_pressure_grid

        grid = build_helios_pressure_grid(
            p_boa_microbar=payload["grid"]["p_boa_microbar"],
            p_toa_microbar=payload["grid"]["p_toa_microbar"],
            n_layers=args.layers,
        )
        write_tp_profile_from_grid(
            args.write_tp,
            grid=grid,
            temperature_boa_k=float(payload["frozen"]["temperature_boa_k"]),
            temperature_lay_k=np.asarray(payload["frozen"]["temperature_lay_k"]),
        )
    print(json.dumps({"out": str(out), "mode": payload["mode"], "n_layers": args.layers}, indent=2))
    return payload


if __name__ == "__main__":
    main()

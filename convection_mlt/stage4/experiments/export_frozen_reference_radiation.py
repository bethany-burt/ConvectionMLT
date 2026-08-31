"""Export frozen Stage-3 radiation reference on nested columns for HELIOS parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import (
    HeliosAdapter,
    LowerNetInternalFlux,
    TopIrradiation,
    nested_analytic_opacity_spec,
    solve_radiation,
)
from convection_mlt.adapters.helios import (
    flux_si_to_cgs,
    heating_from_net_flux,
    pressure_pa_to_microbar,
    write_tp_profile,
)
from convection_mlt.adapters.helios_contracts import (
    F_INT,
    F_IRR,
    GRAVITY_SI,
    HELIOS_DEFAULT_DIFFUSIVITY,
    PINNED_HELIOS_COMMIT,
    PROVENANCE_ONLY,
    STAGE3_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    T_INT,
    TP_PRESSURE_UNIT,
)

RESULTS = ROOT / "results"
NESTED = RESULTS / "nested_rce_family.json"
N192 = RESULTS / "n192_implicit_rce.json"
FIXTURES = ROOT / "fixtures" / "helios"


def _load_record(n_layers: int) -> dict:
    if n_layers == 192:
        return json.loads(N192.read_text())
    members = json.loads(NESTED.read_text()).get("members") or {}
    key = str(n_layers)
    if key not in members:
        raise FileNotFoundError(f"nested member N={n_layers} not in {NESTED}")
    return members[key]


def export_reference(
    n_layers: int,
    *,
    thermal_only: bool,
    diffusivity: float,
) -> dict:
    rec = _load_record(n_layers)
    spec = nested_analytic_opacity_spec(n_layers)
    grid = spec.grid()
    t = np.asarray(rec["temperature"], dtype=np.float64)
    mass = np.asarray(grid.layer_mass, dtype=np.float64)
    p = np.asarray(grid.pressure_centres, dtype=np.float64)
    f_irr = 0.0 if thermal_only else spec.f_irr
    mode = "thermal_only" if thermal_only else "irradiated"
    rad = solve_radiation(
        t,
        mass,
        spec.opacity(),
        p,
        TopIrradiation(f_irr),
        LowerNetInternalFlux(spec.f_int),
        bottom_convective_flux=0.0,
        diffusivity_factor=diffusivity,
    )
    adapter = HeliosAdapter(helios_top_to_bottom=True)
    f_net = np.asarray(rad.flux_net, dtype=np.float64)
    f_up = np.asarray(rad.flux_up.sum(axis=0), dtype=np.float64)
    f_down = np.asarray(rad.flux_down.sum(axis=0), dtype=np.float64)
    heating = heating_from_net_flux(f_net, mass)
    edges = np.asarray(spec.pressure_edges(), dtype=np.float64)
    payload = {
        "purpose": "Stage-3 radiation reference on frozen T(P) for HELIOS radiation-only parity",
        "coupled_helios_rce_claimed": False,
        "comparison_type": "radiation_only_not_mlt",
        "mode": mode,
        "n_layers": n_layers,
        "source_record": "n192_implicit_rce.json" if n_layers == 192 else f"nested_rce_family.json[{n_layers}]",
        "source_status": rec.get("status"),
        "profile_checksum_sha256": rec.get("profile_checksum_sha256"),
        "helios_commit": PINNED_HELIOS_COMMIT,
        "contracts": {
            "gravity_si": GRAVITY_SI,
            "f_int": F_INT,
            "f_irr": f_irr,
            "t_int_K": T_INT,
            "diffusivity_factor": diffusivity,
            "opacity": "nested_analytic_opacity_spec",
            "lower_bc": "LowerNetInternalFlux",
            "bottom_convective_flux": 0.0,
            "canonical_orientation": "bottom_to_top",
            "helios_orientation": "top_to_bottom",
            "tp_pressure_unit": TP_PRESSURE_UNIT,
            "flux_unit_canonical": "W m^-2",
            "flux_unit_helios": "erg s^-1 cm^-2",
            "provenance_only": PROVENANCE_ONLY,
        },
        "frozen": {
            "pressure_centres_Pa": p.tolist(),
            "pressure_edges_Pa": edges.tolist(),
            "pressure_interfaces_helios_microbar": pressure_pa_to_microbar(
                adapter.from_canonical_interfaces(edges)
            ).tolist(),
            "mass_path_kg_m2": mass.tolist(),
            "temperature_K": t.tolist(),
            "flux_up_W_m2": f_up.tolist(),
            "flux_down_W_m2": f_down.tolist(),
            "flux_net_W_m2": f_net.tolist(),
            "heating_W_m2": heating.tolist(),
            "flux_net_helios_erg_s_cm2": flux_si_to_cgs(
                adapter.from_canonical_interfaces(f_net)
            ).tolist(),
            "temperature_helios_K": adapter.from_canonical_layers(t).tolist(),
            "pressure_centres_helios_microbar": pressure_pa_to_microbar(
                adapter.from_canonical_layers(p)
            ).tolist(),
        },
        "diagnostics": {
            "flux_net_bottom": float(f_net[0]),
            "flux_net_top": float(f_net[-1]),
            "bottom_boundary_residual": float(abs(f_net[0] - spec.f_int)),
            "sigma_Tint4": float(STEFAN_BOLTZMANN * T_INT**4),
            "orientation_roundtrip_exact": bool(
                np.array_equal(adapter.roundtrip_layers(t), t)
            ),
        },
        "sensitivity": {
            "stage3_diffusivity_D166": STAGE3_DIFFUSIVITY,
            "helios_default_diffusivity": HELIOS_DEFAULT_DIFFUSIVITY,
        },
    }
    return payload


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, choices=(96, 192), required=True)
    parser.add_argument(
        "--mode",
        choices=("thermal-only", "irradiated"),
        default="thermal-only",
    )
    parser.add_argument(
        "--diffusivity",
        type=float,
        default=HELIOS_DEFAULT_DIFFUSIVITY,
        help="Parity reference diffusivity (HELIOS-equivalent)",
    )
    parser.add_argument("--write-tp", type=Path, default=None)
    args = parser.parse_args()
    thermal = args.mode == "thermal-only"
    payload = export_reference(
        args.layers,
        thermal_only=thermal,
        diffusivity=args.diffusivity,
    )
    suffix = f"n{args.layers}_{'thermal' if thermal else 'irradiated'}"
    out = RESULTS / f"frozen_reference_radiation_{suffix}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.write_tp is not None:
        write_tp_profile(
            args.write_tp,
            temperature_k=np.asarray(payload["frozen"]["temperature_K"]),
            pressure_centres_pa=np.asarray(payload["frozen"]["pressure_centres_Pa"]),
            boa_temperature_k=T_INT,
            boa_pressure_pa=float(payload["frozen"]["pressure_edges_Pa"][0]),
        )
    print(json.dumps({"out": str(out), "mode": payload["mode"], "n_layers": args.layers}, indent=2))
    return payload


if __name__ == "__main__":
    main()

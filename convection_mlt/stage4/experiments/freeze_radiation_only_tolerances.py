"""Freeze offline radiation-only comparison tolerances before any live HELIOS run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import STEFAN_BOLTZMANN, nested_analytic_opacity_spec
from convection_mlt.adapters.helios_contracts import (
    HELIOS_DEFAULT_DIFFUSIVITY,
    MICROBAR_TO_PA,
    OPACITY_SI_TO_CGS,
    PINNED_HELIOS_COMMIT,
    STAGE3_DIFFUSIVITY,
)
from convection_mlt.adapters.helios_grid import (
    build_helios_grid_from_nested_edges,
    layer_optical_depth_cgs,
    layer_optical_depth_si,
    sample_nested_tp_on_helios_grid,
)
from convection_mlt.adapters.helios_opacity_table import (
    analytic_kappa_cgs,
    bolometric_from_table_bands,
    build_table_arrays,
    interpolate_opacity_cgs,
    read_helios_opacity_hdf5,
)
from export_helios_grid_reference import _load_record

FIXTURES = ROOT / "fixtures" / "helios"
TABLE = FIXTURES / "analytic_grey_nested.h5"
OUT = FIXTURES / "radiation_only_tolerances.json"
RESULTS = ROOT / "results"


def _load_temperature_hull(layers: tuple[int, ...]) -> np.ndarray:
    temps = []
    for n in layers:
        rec = _load_record(n)
        temps.append(np.asarray(rec["temperature"], dtype=np.float64))
    return np.concatenate(temps)


def _opacity_interpolation_budget(table_path: Path) -> float:
    if not table_path.exists():
        spec = nested_analytic_opacity_spec(96)
        opacity = spec.opacity()
        table = build_table_arrays(
            opacity, t_min=200.0, t_max=2000.0, p_min_bar=1e-9, p_max_bar=100.0
        )
    else:
        table = read_helios_opacity_hdf5(table_path)
    max_rel = 0.0
    for it in range(1, table.ntemp - 1):
        for ip in range(1, table.npress - 1):
            t0, t1 = float(table.temperatures_k[it]), float(table.temperatures_k[it + 1])
            p0, p1 = float(table.pressures_bar[ip]), float(table.pressures_bar[ip + 1])
            t_off = 0.5 * (t0 + t1)
            p_off = 0.5 * (p0 + p1)
            spec = nested_analytic_opacity_spec(96)
            k_analytic = float(analytic_kappa_cgs(spec.opacity(), t_off, p_off)[0])
            k_tab = interpolate_opacity_cgs(table, t_off, p_off)
            denom = max(abs(k_analytic), 1e-30)
            max_rel = max(max_rel, abs(k_tab - k_analytic) / denom)
    return float(max_rel)


def _kappa_si_cgs_gate(n_layers: int) -> float:
    rec = _load_record(n_layers)
    spec = nested_analytic_opacity_spec(n_layers)
    edges = np.asarray(rec["pressure_edges"], dtype=np.float64)
    grid = build_helios_grid_from_nested_edges(edges, n_layers)
    _, t_lay = sample_nested_tp_on_helios_grid(rec, grid)
    op = spec.opacity()
    k_si = op.evaluate(t_lay, grid.p_lay_Pa)[0]
    k_cgs = k_si * OPACITY_SI_TO_CGS
    k_direct = analytic_kappa_cgs(op, t_lay, grid.p_lay_Pa / 1.0e5)
    rel = np.max(np.abs(k_cgs - k_direct) / np.maximum(np.abs(k_direct), 1e-30))
    return float(rel)


def _delta_tau_gate(n_layers: int) -> float:
    rec = _load_record(n_layers)
    spec = nested_analytic_opacity_spec(n_layers)
    edges = np.asarray(rec["pressure_edges"], dtype=np.float64)
    grid = build_helios_grid_from_nested_edges(edges, n_layers)
    _, t_lay = sample_nested_tp_on_helios_grid(rec, grid)
    op = spec.opacity()
    k_si = op.evaluate(t_lay, grid.p_lay_Pa)[0]
    k_cgs = k_si * OPACITY_SI_TO_CGS
    dp_pa = np.abs(np.diff(grid.p_int_microbar)) * MICROBAR_TO_PA
    dp_micro = np.abs(np.diff(grid.p_int_microbar))
    dt_si = layer_optical_depth_si(k_si, dp_pa)
    dt_cgs = layer_optical_depth_cgs(k_cgs, dp_micro)
    rel = np.max(np.abs(dt_si - dt_cgs) / np.maximum(dt_si, 1e-30))
    return float(rel)


def _planck_bolometric_gate(table_path: Path, t_hull: np.ndarray) -> float:
    if not table_path.exists():
        spec = nested_analytic_opacity_spec(96)
        table = build_table_arrays(
            spec.opacity(), t_min=200.0, t_max=2000.0, p_min_bar=1e-9, p_max_bar=100.0
        )
    else:
        table = read_helios_opacity_hdf5(table_path)
    sigma_t4 = STEFAN_BOLTZMANN * t_hull**4
    planck = bolometric_from_table_bands(t_hull, table.wavelengths_cm)
    return float(np.max(np.abs(planck - sigma_t4) / np.maximum(sigma_t4, 1e-30)))


def main() -> dict:
    layers = (96, 192)
    t_hull = _load_temperature_hull(layers)
    adapter_roundoff = 1.0e-15
    opacity_interp = _opacity_interpolation_budget(TABLE)
    kappa_si_cgs = max(_kappa_si_cgs_gate(96), _kappa_si_cgs_gate(192))
    delta_tau = max(_delta_tau_gate(96), _delta_tau_gate(192))
    planck_rel = _planck_bolometric_gate(TABLE, t_hull)
    closure_note = (
        "Parity reference uses HELIOS-equivalent diffusivity; "
        f"D={STAGE3_DIFFUSIVITY} archived as sensitivity only."
    )
    closure_remainder = (
        "configured_in_reference_not_subtracted"
        if HELIOS_DEFAULT_DIFFUSIVITY != STAGE3_DIFFUSIVITY
        else 0.0
    )

    tol_flux_net = max(adapter_roundoff, opacity_interp, planck_rel, delta_tau, 1.0e-8)
    tol_heating = tol_flux_net
    # HELIOS writes integrated_flux.dat with default %g (~6 significant digits).
    # Independently rounded F↑, F↓, F_net cannot satisfy F_net = F↑ − F↓ to 1e-12.
    serialization_text_rel = 1.0e-4
    tol_boundary = max(adapter_roundoff, serialization_text_rel)
    tol_grid = max(adapter_roundoff, 1.0e-6)
    tol_toa = max(adapter_roundoff, serialization_text_rel)
    tol_decomp = max(serialization_text_rel, adapter_roundoff)
    identity_rel = 1.0e-12

    payload = {
        "frozen_before_live": True,
        "helios_commit": PINNED_HELIOS_COMMIT,
        "derivation": {
            "adapter_roundoff_rel": adapter_roundoff,
            "opacity_interpolation_rel": opacity_interp,
            "kappa_si_cgs_rel": kappa_si_cgs,
            "delta_tau_si_cgs_rel": delta_tau,
            "planck_bolometric_rel": planck_rel,
            "closure_remainder": closure_remainder,
            "closure_note": closure_note,
            "helios_text_serialization_rel": serialization_text_rel,
            "parity_diffusivity": HELIOS_DEFAULT_DIFFUSIVITY,
            "sensitivity_diffusivity": STAGE3_DIFFUSIVITY,
        },
        "gates": {
            "pressure_grid_rel": tol_grid,
            "f_intern_parameter_rel": tol_boundary,
            "flux_decomposition_rel": tol_decomp,
            "rocky_surface_up_rel": tol_boundary,
            "rocky_surface_net_rel": tol_boundary,
            "toa_flux_down_rel": tol_toa,
            "flux_up_rel": tol_flux_net,
            "flux_down_rel": tol_flux_net,
            "flux_net_rel": tol_flux_net,
            "heating_from_flux_rel": tol_heating,
            "reference_column_energy_identity_rel": identity_rel,
            "helios_column_energy_closure_rel": serialization_text_rel,
            "column_energy_closure_rel": identity_rel,
        },
        "comparison_order": [
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
        "offline_blocking_gates": {
            "kappa_si_cgs": kappa_si_cgs,
            "delta_tau_si_cgs": delta_tau,
            "planck_bolometric": planck_rel,
            "opacity_interpolation": opacity_interp,
        },
        "note": (
            "Tolerances derived offline on HELIOS grid. No live HELIOS residual used. "
            "Parity flux/heating gates are interpolation-budget; text-file contracts "
            "use HELIOS %g serialization tolerances. Do not call this immediately "
            "before scoring a live run."
        ),
    }
    return payload


def write_tolerances(payload: dict, *, force: bool = False) -> None:
    if OUT.exists() and not force:
        raise SystemExit(
            f"Refusing to overwrite frozen tolerances at {OUT}. "
            "Pass --force only for an offline re-derivation, never on a live scoring path."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = main()
    write_tolerances(payload, force=args.force)
    print(json.dumps({"out": str(OUT), "flux_net_tol": payload["gates"]["flux_net_rel"]}, indent=2))

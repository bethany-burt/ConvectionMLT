from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from convection_mlt import (
    ConstantGravity,
    ConstantGreyOpacity,
    ConstantH2Thermo,
    HeliosAdapter,
    LowerNetInternalFlux,
    PhysicsConfig,
    RCEConfig,
    RCERoute,
    SolverConfig,
    TopIrradiation,
    build_grid,
    load_integrated_flux,
    load_tp_profile,
    log_pressure_edges,
    solve_adaptive_rce,
)

PINNED_HELIOS_COMMIT = "b0800f9ea4366263241c13bb926e8ca68f266cc5"
HELIOS_T_GATE = 1.0e-8
HELIOS_FLUX_GATE = 1.0e-8
CONV_OFF_FLUX_GATE = 1.0e-8


def _normalize_diff(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    n = min(a.size, b.size)
    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    scale = np.maximum.reduce([np.abs(aa), np.abs(bb), np.full(n, floor)])
    return float(np.max(np.abs(aa - bb) / scale, initial=0.0))


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_mlt_reference(n_layers: int, *, convection_off: bool = False):
    gravity = 15.0
    grid = build_grid(log_pressure_edges(5.0e6, 1.0e2, n_layers), gravity)
    p = grid.pressure_centres
    t0 = 900.0 * (p / p[0]) ** 0.58
    thermo = ConstantH2Thermo()
    physics = PhysicsConfig(
        gravity=gravity,
        alpha=0.0 if convection_off else 1.0,
        closure_prefactor=0.5,
    )
    solver = SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)
    cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10)
    res = solve_adaptive_rce(
        grid, t0, physics, solver, thermo, ConstantGreyOpacity(2.0e-4), p,
        TopIrradiation(flux=120.0), LowerNetInternalFlux(flux=300.0),
        gravity=ConstantGravity(gravity),
        route=RCERoute.UNSPLIT,
        config=cfg,
    )
    return res.final_state.temperature, res.final_flux_total, res.primary_rcb_log10p, res.status.value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage 4 MLT RCE against a live HELIOS output bundle.")
    parser.add_argument("--helios-output-dir", type=Path, required=True)
    parser.add_argument("--helios-case", type=str, required=True)
    parser.add_argument("--helios-commit", type=str, default=PINNED_HELIOS_COMMIT)
    parser.add_argument("--helios-conv-off-flux-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("stage4/results/live_helios_comparison.json"))
    args = parser.parse_args()

    if args.helios_commit != PINNED_HELIOS_COMMIT:
        raise ValueError(
            f"HELIOS commit mismatch: expected {PINNED_HELIOS_COMMIT}, got {args.helios_commit}"
        )

    tp_file = args.helios_output_dir / f"{args.helios_case}_tp.dat"
    flux_file = args.helios_output_dir / f"{args.helios_case}_integrated_flux.dat"
    if not tp_file.exists() or not flux_file.exists():
        raise FileNotFoundError("Missing HELIOS output files (_tp.dat and _integrated_flux.dat)")

    tp = load_tp_profile(tp_file)
    flux = load_integrated_flux(flux_file)
    adapter = HeliosAdapter(legacy_reverse=True)

    helios_t_layers = tp.temperature_k[tp.layer_index >= 0]
    helios_t_canonical = adapter.to_canonical_layers(helios_t_layers)
    helios_fnet_si = adapter.to_canonical_interfaces(flux.flux_net_cgs) * 1.0e-3

    mlt_t, mlt_fnet, mlt_rcb, mlt_status = _run_mlt_reference(helios_t_canonical.size)
    temp_err = _normalize_diff(mlt_t, helios_t_canonical, 1e-9)
    flux_err = _normalize_diff(mlt_fnet, helios_fnet_si, 1e-12)

    orientation_roundtrip_ok = bool(np.array_equal(adapter.roundtrip_layers(helios_t_layers), helios_t_layers))

    conv_off_rel = None
    conv_off_scale = None
    conv_off_finite = None
    conv_off_checksum = _sha256_file(args.helios_conv_off_flux_file)
    if args.helios_conv_off_flux_file is not None:
        conv_off_flux = load_integrated_flux(args.helios_conv_off_flux_file)
        conv_off_net = adapter.to_canonical_interfaces(conv_off_flux.flux_net_cgs) * 1.0e-3
        conv_off_finite = bool(np.all(np.isfinite(conv_off_net)))
        _t_off, mlt_off_flux, _rcb, _st = _run_mlt_reference(helios_t_canonical.size, convection_off=True)
        conv_off_scale = float(max(np.max(np.abs(mlt_off_flux)), np.max(np.abs(conv_off_net)), 1e-12))
        conv_off_rel = _normalize_diff(mlt_off_flux, conv_off_net, 1e-12)

    t_pass = temp_err <= HELIOS_T_GATE
    f_pass = flux_err <= HELIOS_FLUX_GATE
    conv_off_pass = conv_off_rel is not None and conv_off_rel <= CONV_OFF_FLUX_GATE
    status = "pass" if (orientation_roundtrip_ok and t_pass and f_pass and conv_off_pass) else "pilot"

    result = {
        "helios_commit": args.helios_commit,
        "helios_case": args.helios_case,
        "status": status,
        "note": (
            "Live HELIOS executable completed. Differences near unity are not parity; "
            "MLT vs HELIOS used unmatched opacity/BCs. Treat as a pilot."
        ),
        "matched_setup": False,
        "metrics": {
            "equilibrium_temperature_max_rel": temp_err,
            "equilibrium_temperature_tolerance": HELIOS_T_GATE,
            "equilibrium_temperature_status": "PASS" if t_pass else "FAIL",
            "equilibrium_flux_total_max_rel": flux_err,
            "equilibrium_flux_total_tolerance": HELIOS_FLUX_GATE,
            "equilibrium_flux_total_status": "PASS" if f_pass else "FAIL",
            "primary_rcb_log10p_mlt": mlt_rcb,
            "mlt_terminal_status": mlt_status,
        },
        "point_38": {
            "orientation_roundtrip_exact": orientation_roundtrip_ok,
            "convection_off_flux_file_used": str(args.helios_conv_off_flux_file) if args.helios_conv_off_flux_file else None,
            "convection_off_flux_file_portable": args.helios_conv_off_flux_file.name if args.helios_conv_off_flux_file else None,
            "convection_off_checksum_sha256": conv_off_checksum,
            "convection_off_finite_flux": conv_off_finite,
            "convection_off_flux_max_rel": conv_off_rel,
            "convection_off_flux_scale": conv_off_scale,
            "convection_off_flux_tolerance": CONV_OFF_FLUX_GATE,
            "convection_off_flux_status": (
                "PASS" if conv_off_pass else "FAIL" if conv_off_rel is not None else "pending"
            ),
            "conv_on_flux_file_portable": flux_file.name,
            "conv_on_checksum_sha256": _sha256_file(flux_file),
            "tp_file_portable": tp_file.name,
            "tp_checksum_sha256": _sha256_file(tp_file),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

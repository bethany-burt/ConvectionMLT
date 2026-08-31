"""Offline RCB discrepancy diagnostics for coupled HELIOS N=96 (job 16015698).

1. Matched Stage-3 radiation on the final HELIOS T(P): same grid, opacity, D=2,
   F↓_TOA=0, and HELIOS F↑_BOA (not F_int=300). Compare F↑/F↓/F_rad,net/ΔF_rad.
2. Cross convective-stability: MLT instability on HELIOS T, HELIOS-like lapse on MLT T.

Does not rerun HELIOS. Does not claim Stage-4 headline.
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

from compare_coupled_helios_rce import load_mlt_reference
from convection_mlt import (
    ConstantH2Thermo,
    LowerUpwardFlux,
    STEFAN_BOLTZMANN,
    TopIrradiation,
    build_grid,
    load_integrated_flux,
    mixing_length_flux,
    nested_analytic_opacity_spec,
    solve_radiation,
    to_canonical_interfaces,
)
from convection_mlt.adapters.helios import (
    flux_cgs_to_si,
    layer_energy_increment,
    load_tp_profile,
)
from convection_mlt.adapters.helios_contracts import (
    GRAVITY_SI,
    HELIOS_DEFAULT_DIFFUSIVITY,
    MICROBAR_TO_PA,
    PROVENANCE_ONLY,
)
from convection_mlt.adapters.helios_grid import (
    build_helios_pressure_grid,
    interpolate_log_pressure,
)

RESULTS = ROOT / "results"
DEBUG = RESULTS / "helios_coupled_n96_job16015698_debug" / "iterative"
NABLA_AD = float(PROVENANCE_ONLY["nabla_ad"])
DEFAULT_ALPHA = 1.0
# Radiation-only opacity-interpolation gate (not a Stage-4 headline).
MATCHED_FLUX_REL_GATE = 0.005278229379031692


def _helios_layers(tp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lay = tp.layer_index != -1
    p = np.asarray(tp.pressure_microbar[lay], dtype=np.float64) * MICROBAR_TO_PA
    t = np.asarray(tp.temperature_k[lay], dtype=np.float64)
    flag_u = np.asarray(tp.conv_unstable_flag[lay], dtype=np.float64)
    flag_l = np.asarray(tp.conv_lapse_flag[lay], dtype=np.float64)
    if np.any(flag_u > 0.5):
        flag = flag_u
    else:
        flag = flag_l
    return p, t, flag


def _helios_boa_temperature_k(tp) -> float:
    boa = tp.layer_index == -1
    if not np.any(boa):
        raise ValueError("HELIOS tp.dat missing BOA row")
    return float(np.asarray(tp.temperature_k[boa], dtype=np.float64)[0])


def _norm_diff(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = min(aa.size, bb.size)
    scale = np.maximum.reduce([np.abs(aa[:n]), np.abs(bb[:n]), np.full(n, floor)])
    return float(np.max(np.abs(aa[:n] - bb[:n]) / scale))


def stage3_radiation_on_helios_tp(
    tp_path: Path,
    flux_path: Path,
    n_layers: int = 96,
) -> dict:
    """Matched Stage-3 RT on final HELIOS T(P); never impose F_int as F↑_BOA."""
    tp = load_tp_profile(tp_path)
    flux = load_integrated_flux(flux_path)
    p_lay, t_lay, flag = _helios_layers(tp)
    t_boa = _helios_boa_temperature_k(tp)
    rec = load_mlt_reference(n_layers)
    grid = build_helios_pressure_grid(
        p_boa_microbar=float(rec["helios_p_boa_microbar"]),
        p_toa_microbar=float(rec["helios_p_toa_microbar"]),
        n_layers=n_layers,
    )
    p_rel = np.max(np.abs(grid.p_lay_Pa - p_lay) / np.maximum(grid.p_lay_Pa, p_lay))
    if p_rel < 1.0e-5 and t_lay.size == n_layers:
        t_on = np.asarray(t_lay, dtype=np.float64).copy()
        t_source = "native_helios_layer_centres"
    else:
        t_on = interpolate_log_pressure(p_lay, t_lay, grid.p_lay_Pa)
        t_source = "logP_interp_to_helios_grid"

    helios_up = to_canonical_interfaces(
        flux_cgs_to_si(flux.flux_up_cgs), flux.pressure_microbar, n_layers=n_layers
    )
    helios_down = to_canonical_interfaces(
        flux_cgs_to_si(flux.flux_down_cgs), flux.pressure_microbar, n_layers=n_layers
    )
    helios_net = to_canonical_interfaces(
        flux_cgs_to_si(flux.flux_net_cgs), flux.pressure_microbar, n_layers=n_layers
    )
    f_up_boa = float(helios_up[0])
    sigma_tboa4 = float(STEFAN_BOLTZMANN * t_boa**4)

    spec = nested_analytic_opacity_spec(n_layers)
    opacity = spec.opacity()
    rad = solve_radiation(
        t_on,
        grid.layer_mass_kg_m2,
        opacity,
        grid.p_lay_Pa,
        TopIrradiation(0.0),
        LowerUpwardFlux(f_up_boa),
        diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
    )
    s3_up = np.sum(np.asarray(rad.flux_up, dtype=np.float64), axis=0)
    s3_down = np.sum(np.asarray(rad.flux_down, dtype=np.float64), axis=0)
    s3_net = np.asarray(rad.flux_net, dtype=np.float64)
    s3_df = layer_energy_increment(s3_net)
    h_df = layer_energy_increment(helios_net)
    f_floor = max(abs(f_up_boa), abs(float(helios_net[0])), 1.0)

    flux_up_rel = _norm_diff(s3_up, helios_up, f_floor)
    flux_down_rel = _norm_diff(s3_down, helios_down, f_floor)
    flux_net_rel = _norm_diff(s3_net, helios_net, f_floor)
    dF_rad_rel = _norm_diff(s3_df, h_df, f_floor)
    matched_agree = bool(
        max(flux_up_rel, flux_down_rel, flux_net_rel, dF_rad_rel) <= MATCHED_FLUX_REL_GATE
    )

    # Adiabat join on native HELIOS layers (authoritative for RCB on this T).
    logp = np.log(p_lay)
    logt = np.log(np.maximum(t_lay, 1.0))
    nabla = (logt[:-1] - logt[1:]) / (logp[:-1] - logp[1:])
    on_adiabat = nabla >= (NABLA_AD - 1.0e-3)
    rcb_log10p = None
    if np.any(on_adiabat) and bool(on_adiabat[0]):
        i_hi = 0
        while i_hi + 1 < on_adiabat.size and on_adiabat[i_hi + 1]:
            i_hi += 1
        rcb_log10p = float(np.log10(p_lay[i_hi]))
    helios_unstable = flag > 0.5
    helios_rcb = None
    if np.any(helios_unstable) and bool(helios_unstable[0]):
        i_hi = 0
        while i_hi + 1 < helios_unstable.size and helios_unstable[i_hi + 1]:
            i_hi += 1
        helios_rcb = float(np.log10(p_lay[i_hi]))
    n_cz = int(np.sum(helios_unstable))
    return {
        "purpose": (
            "Matched Stage-3 radiation on final HELIOS T(P); compare F↑/F↓/F_rad,net/ΔF_rad"
        ),
        "diffusivity_factor": HELIOS_DEFAULT_DIFFUSIVITY,
        "f_irr": 0.0,
        "temperature_source": t_source,
        "max_layer_pressure_rel": float(p_rel),
        "lower_bc": {
            "type": "LowerUpwardFlux",
            "f_up_boa_W_m2": f_up_boa,
            "helios_boa_temperature_k": t_boa,
            "sigma_Tboa4_W_m2": sigma_tboa4,
            "sigma_Tboa4_vs_Fup_rel": abs(sigma_tboa4 - f_up_boa) / max(abs(f_up_boa), 1.0),
            "note": (
                "F↑_BOA is HELIOS surface emission, not F_int=300. "
                "Prior bug: np.interp on descending layer P extrapolated a bogus "
                "T_boa≈227 K → σT⁴≈150 W m⁻² and scrambled layer T."
            ),
        },
        "helios_column": {
            "f_up_boa_W_m2": float(helios_up[0]),
            "f_down_boa_W_m2": float(helios_down[0]),
            "f_rad_net_boa_W_m2": float(helios_net[0]),
            "f_up_toa_W_m2": float(helios_up[-1]),
            "f_down_toa_W_m2": float(helios_down[-1]),
            "f_rad_net_toa_W_m2": float(helios_net[-1]),
        },
        "stage3_column": {
            "f_up_boa_W_m2": float(s3_up[0]),
            "f_down_boa_W_m2": float(s3_down[0]),
            "f_rad_net_boa_W_m2": float(s3_net[0]),
            "f_up_toa_W_m2": float(s3_up[-1]),
            "f_down_toa_W_m2": float(s3_down[-1]),
            "f_rad_net_toa_W_m2": float(s3_net[-1]),
        },
        "matched_metrics": {
            "flux_up_rel": flux_up_rel,
            "flux_down_rel": flux_down_rel,
            "flux_net_rel": flux_net_rel,
            "dF_rad_rel": dF_rad_rel,
            "gate_rel": MATCHED_FLUX_REL_GATE,
            "agree": matched_agree,
        },
        "decision_tree": (
            "If matched radiation agrees: RCB shift is convection closure and/or grid. "
            "If it disagrees: quantify radiation/source treatment before like-for-like claims."
        ),
        "stage3_rcb_log10p_from_adiabat_join": rcb_log10p,
        "helios_rcb_log10p_from_flags": helios_rcb,
        "rcb_dex_stage3_vs_helios_flags": (
            None if rcb_log10p is None or helios_rcb is None else abs(rcb_log10p - helios_rcb)
        ),
        "n_near_adiabatic_interfaces": int(np.sum(on_adiabat)),
        "n_cz_helios_layers": n_cz,
        "mean_nabla_in_flagged_cz": (
            float(np.mean(nabla[: max(n_cz - 1, 0)])) if n_cz > 1 else None
        ),
        "nabla_ad": NABLA_AD,
        "untrustworthy_prior_f_net_toa_W_m2": 149.99562134482326,
        "note": (
            "Do not compare Stage-3 F_rad to F_int=300. Prior ~150 W m⁻² TOA result "
            "was an unmatched lower-BC / descending-P interp bug, not HELIOS physics."
        ),
    }


def cross_convective_stability(tp_path: Path, n_layers: int = 96) -> dict:
    tp = load_tp_profile(tp_path)
    p_h, t_h, flag_h = _helios_layers(tp)
    rec = load_mlt_reference(n_layers)
    p_m = np.asarray(rec["pressure_centres"], dtype=np.float64)
    t_m = np.asarray(rec["temperature"], dtype=np.float64)
    alpha = float(rec.get("alpha") or rec.get("mixing_length_alpha") or DEFAULT_ALPHA)

    grid_h = build_helios_pressure_grid(
        p_boa_microbar=float(rec["helios_p_boa_microbar"]),
        p_toa_microbar=float(rec["helios_p_toa_microbar"]),
        n_layers=n_layers,
    )
    pg = build_grid(grid_h.p_int_Pa, GRAVITY_SI)
    thermo = ConstantH2Thermo()
    t_on_h = interpolate_log_pressure(p_h, t_h, pg.pressure_centres)
    cl_h = mixing_length_flux(pg, t_on_h, GRAVITY_SI, alpha, thermo)
    # Compute adiabat join on the native HELIOS layer centres (not remapped grid).
    logp_h = np.log(p_h)
    logt_h = np.log(np.maximum(t_h, 1.0))
    nabla_h = (logt_h[:-1] - logt_h[1:]) / (logp_h[:-1] - logp_h[1:])
    on_ad_h = nabla_h >= (NABLA_AD - 1.0e-3)
    mlt_unstable_on_helios = np.asarray(cl_h.superadiabaticity[1:-1] > 0.0, dtype=bool)
    p_iface = pg.pressure_edges[1:-1]
    helios_cz_layers = flag_h > 0.5

    logp_m = np.log(p_m)
    logt_m = np.log(np.maximum(t_m, 1.0))
    nabla_m = (logt_m[:-1] - logt_m[1:]) / (logp_m[:-1] - logp_m[1:])
    helios_like_on_mlt = nabla_m > NABLA_AD
    on_ad_m = nabla_m >= (NABLA_AD - 1.0e-3)

    def _rcb_from_mask(mask: np.ndarray, p: np.ndarray) -> float | None:
        if not np.any(mask) or not bool(mask[0]):
            return None
        i_hi = 0
        while i_hi + 1 < mask.size and mask[i_hi + 1]:
            i_hi += 1
        return float(np.log10(float(p[min(i_hi, p.size - 1)])))

    rcb_mlt_ref = rec.get("primary_rcb_log10p")
    rcb_mlt_sa_on_helios = _rcb_from_mask(mlt_unstable_on_helios, p_iface)
    rcb_adiabat_join_helios_T = _rcb_from_mask(on_ad_h, p_h[:-1])
    rcb_helios_flags = _rcb_from_mask(helios_cz_layers, p_h)
    rcb_helios_like_on_mlt = _rcb_from_mask(helios_like_on_mlt, p_m[:-1])
    rcb_adiabat_join_mlt_T = _rcb_from_mask(on_ad_m, p_m[:-1])

    n_cmp = min(helios_cz_layers.size - 1, on_ad_h.size)
    agree = int(np.sum(helios_cz_layers[:n_cmp] == on_ad_h[:n_cmp]))

    return {
        "purpose": "Cross convective-stability diagnostic for RCB discrepancy",
        "alpha": alpha,
        "mlt_reference_rcb_log10p": rcb_mlt_ref,
        "mlt_superadiabatic_on_helios_T": {
            "rcb_log10p": rcb_mlt_sa_on_helios,
            "n_unstable": int(np.sum(mlt_unstable_on_helios)),
            "note": (
                "Expected near zero for HELIOS convective-adjustment T (already "
                "driven to ∇≈∇ad)."
            ),
        },
        "adiabat_join_on_helios_T": {
            "rcb_log10p": rcb_adiabat_join_helios_T,
            "n_near_adiabatic": int(np.sum(on_ad_h)),
            "dex_vs_mlt_ref": (
                None
                if rcb_adiabat_join_helios_T is None or rcb_mlt_ref is None
                else abs(float(rcb_adiabat_join_helios_T) - float(rcb_mlt_ref))
            ),
            "dex_vs_helios_flags": (
                None
                if rcb_adiabat_join_helios_T is None or rcb_helios_flags is None
                else abs(float(rcb_adiabat_join_helios_T) - float(rcb_helios_flags))
            ),
        },
        "helios_lapse_flags_on_helios_T": {
            "rcb_log10p": rcb_helios_flags,
            "n_cz_layers": int(np.sum(helios_cz_layers)),
            "dex_vs_mlt_ref": (
                None
                if rcb_helios_flags is None or rcb_mlt_ref is None
                else abs(float(rcb_helios_flags) - float(rcb_mlt_ref))
            ),
        },
        "helios_like_superadiabatic_on_mlt_T": {
            "rcb_log10p": rcb_helios_like_on_mlt,
            "n_unstable_interfaces": int(np.sum(helios_like_on_mlt)),
            "dex_vs_mlt_ref": (
                None
                if rcb_helios_like_on_mlt is None or rcb_mlt_ref is None
                else abs(float(rcb_helios_like_on_mlt) - float(rcb_mlt_ref))
            ),
        },
        "adiabat_join_on_mlt_T": {
            "rcb_log10p": rcb_adiabat_join_mlt_T,
            "n_near_adiabatic": int(np.sum(on_ad_m)),
            "dex_vs_mlt_ref": (
                None
                if rcb_adiabat_join_mlt_T is None or rcb_mlt_ref is None
                else abs(float(rcb_adiabat_join_mlt_T) - float(rcb_mlt_ref))
            ),
        },
        "flag_vs_adiabat_join_on_helios_T": {
            "n_agree": agree,
            "n_compared": n_cmp,
            "fraction": agree / max(n_cmp, 1),
        },
        "interpretation_hint": (
            "If adiabat-join on HELIOS T matches HELIOS lapse flags and both sit "
            "~0.3 dex above the MLT RCB, the shift is in the HELIOS T structure "
            "(radiation/source / exact adjustment), not in the instability "
            "diagnostic applied to a shared profile."
        ),
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=Path, default=DEBUG / "final_tp.dat")
    parser.add_argument(
        "--flux",
        type=Path,
        default=DEBUG / "final_integrated_flux.dat",
    )
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "helios_coupled_n96_rcb_diagnostics.json",
    )
    args = parser.parse_args()
    if not args.tp.exists():
        raise SystemExit(f"missing HELIOS TP {args.tp}")
    if not args.flux.exists():
        raise SystemExit(f"missing HELIOS flux {args.flux}")
    stage3 = stage3_radiation_on_helios_tp(args.tp, args.flux, args.layers)
    payload = {
        "job_id": "16015698",
        "helios_tp": str(args.tp),
        "helios_flux": str(args.flux),
        "stage3_radiation_on_helios_tp": stage3,
        "cross_convective_stability": cross_convective_stability(args.tp, args.layers),
        "helios_parity_headline": False,
        "full_stage4_claim": False,
        "note": (
            "Offline only; N=192 postponed until RCB discrepancy is classified. "
            "Matched Stage-3 radiation uses HELIOS F↑_BOA, not F_int."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.output),
        "matched_radiation_agree": stage3["matched_metrics"]["agree"],
        "matched_metrics": stage3["matched_metrics"],
        "stage3_f_rad_net_toa": stage3["stage3_column"]["f_rad_net_toa_W_m2"],
        "helios_f_rad_net_toa": stage3["helios_column"]["f_rad_net_toa_W_m2"],
        "stage3_adiabat_join": stage3.get("stage3_rcb_log10p_from_adiabat_join"),
        "helios_rcb": stage3.get("helios_rcb_log10p_from_flags"),
        "adiabat_join_helios_T": payload["cross_convective_stability"][
            "adiabat_join_on_helios_T"
        ],
        "helios_flags": payload["cross_convective_stability"][
            "helios_lapse_flags_on_helios_T"
        ],
        "helios_like_on_mlt": payload["cross_convective_stability"][
            "helios_like_superadiabatic_on_mlt_T"
        ],
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()

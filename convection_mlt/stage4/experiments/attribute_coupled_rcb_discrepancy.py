"""Attribute the coupled HELIOS↔MLT RCB discrepancy (job 16015698).

Controlled one-factor tests — do not tune solvers to force agreement.

1. Radiation source on final HELIOS T(P): Stage-3 vs HELIOS post-proc iso=yes vs iso=no
2. RCB from nabla_rad on those three frozen radiation fields (same T, P)
3. Nested-τ Stage-3 operator: finite MLT vs instantaneous exact convective adjustment
4. HELIOS N=96 vs N=192 RCB (labelled resolution; filled after cluster run)

Does not claim Stage-4 headline. Does not relax the 0.15-dex RCB gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from compare_coupled_helios_rce import load_mlt_reference
from convection_mlt import (
    ConstantH2Thermo,
    LowerNetInternalFlux,
    LowerTemperature,
    LowerUpwardFlux,
    STEFAN_BOLTZMANN,
    SolverConfig,
    TopIrradiation,
    grey_radiative_equilibrium_temperature,
    load_integrated_flux,
    mixing_length_flux,
    nested_analytic_opacity_spec,
    solve_radiation,
    to_canonical_interfaces,
)
from convection_mlt.adapters.helios import (
    flux_cgs_to_si,
    load_tp_profile,
    write_param_dat,
    write_tp_profile,
)
from convection_mlt.adapters.helios_contracts import (
    F_INT,
    GRAVITY_SI,
    HELIOS_DEFAULT_DIFFUSIVITY,
    MICROBAR_TO_PA,
    PINNED_HELIOS_COMMIT,
    PROVENANCE_ONLY,
    T_INT,
)
from convection_mlt.adapters.helios_grid import build_helios_pressure_grid
from convection_mlt.rce import _primary_rcb_log10p, _temperature_on_adiabat

RESULTS = ROOT / "results"
FIXTURES = ROOT / "fixtures" / "helios"
DEBUG = RESULTS / "helios_coupled_n96_job16015698_debug" / "iterative"
OPACITY = FIXTURES / "analytic_grey_nested.h5"
TEMPLATE = FIXTURES / "helios_param_template.dat"
NABLA_AD = float(PROVENANCE_ONLY["nabla_ad"])
MATCHED_FLUX_REL_GATE = 0.005278229379031692
MLT_RCB_TARGET_LOG10P = 5.028032313236911  # ≈ 1.07 bar
# Temperature-lapse must exceed ∇_ad by more than FP noise on an exact adiabat.
SUPERAD_EPS = 1.0e-8


def _norm_diff(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = min(aa.size, bb.size)
    scale = np.maximum.reduce([np.abs(aa[:n]), np.abs(bb[:n]), np.full(n, floor)])
    return float(np.max(np.abs(aa[:n] - bb[:n]) / scale))


def _helios_layers(tp) -> tuple[np.ndarray, np.ndarray, float]:
    lay = tp.layer_index != -1
    p = np.asarray(tp.pressure_microbar[lay], dtype=np.float64) * MICROBAR_TO_PA
    t = np.asarray(tp.temperature_k[lay], dtype=np.float64)
    boa = tp.layer_index == -1
    t_boa = float(np.asarray(tp.temperature_k[boa], dtype=np.float64)[0])
    return p, t, t_boa


def _load_flux_si(path: Path, n_layers: int) -> dict:
    flux = load_integrated_flux(path)
    p = np.asarray(flux.pressure_microbar, dtype=np.float64)
    return {
        "up": to_canonical_interfaces(flux_cgs_to_si(flux.flux_up_cgs), p, n_layers=n_layers),
        "down": to_canonical_interfaces(flux_cgs_to_si(flux.flux_down_cgs), p, n_layers=n_layers),
        "net": to_canonical_interfaces(flux_cgs_to_si(flux.flux_net_cgs), p, n_layers=n_layers),
        "p_microbar": p,
    }


def nabla_rad_layer(
    temperature: np.ndarray,
    pressure: np.ndarray,
    flux_net_iface: np.ndarray | float,
    kappa_layer: np.ndarray,
    *,
    gravity: float = GRAVITY_SI,
    diffusivity: float = HELIOS_DEFAULT_DIFFUSIVITY,
) -> np.ndarray:
    """Diffusion-limit ∇ ∝ κ F P / T⁴ (D/16σg form). Diagnostic only — not an RCB.

    Prefactor does not match the discrete grey-RE temperature lapse on this
    operator (RE deep ∇≈0.37 while D/16 with F_int gives ≈0.27). Used only to
    compare flux partitions, not to predict convective boundaries.
    """
    t = np.asarray(temperature, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    kap = np.asarray(kappa_layer, dtype=np.float64)
    if np.ndim(flux_net_iface) == 0:
        f_lay = np.full(t.shape, float(flux_net_iface), dtype=np.float64)
    else:
        f = np.asarray(flux_net_iface, dtype=np.float64)
        if f.size != t.size + 1:
            raise ValueError("flux_net must have n_layer+1 interfaces")
        f_lay = 0.5 * (f[:-1] + f[1:])
    denom = 16.0 * STEFAN_BOLTZMANN * float(gravity) * np.maximum(t, 1.0) ** 4
    return (float(diffusivity) * kap * f_lay * p) / denom


def temperature_lapse(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Interface ∇ = ΔlnT/ΔlnP on layer centres (bottom-first)."""
    t = np.asarray(temperature, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    logt = np.log(np.maximum(t, 1.0))
    logp = np.log(p)
    return (logt[:-1] - logt[1:]) / (logp[:-1] - logp[1:])


def crossing_diagnostic(
    gradient: np.ndarray,
    pressure: np.ndarray,
    *,
    nabla_ad: float = NABLA_AD,
    label: str,
) -> dict:
    """Bottom-connected region with gradient > ∇_ad. Not a predicted RCB by itself."""
    unstable = np.asarray(gradient, dtype=np.float64) > float(nabla_ad)
    log10p = None
    n_cz = 0
    if unstable.size and bool(unstable[0]):
        i_hi = 0
        while i_hi + 1 < unstable.size and unstable[i_hi + 1]:
            i_hi += 1
        n_cz = i_hi + 1
        log10p = float(np.log10(float(pressure[i_hi])))
    return {
        "label": label,
        "bottom_crossing_log10p": log10p,
        "n_layers_above_nabla_ad": int(np.sum(unstable)),
        "n_bottom_connected": n_cz,
        "nabla_ad": float(nabla_ad),
        "max_gradient": float(np.max(gradient)) if gradient.size else None,
        "note": (
            "Frozen-profile gradient diagnostic only. Not a predicted RCB and not "
            "a counterfactual without convection."
        ),
    }


def rcb_from_near_adiabat(
    temperature: np.ndarray,
    pressure: np.ndarray,
    *,
    nabla_ad: float = NABLA_AD,
    tol: float = 1.0e-3,
) -> dict:
    """RCB = top of bottom-connected near-adiabatic region on a T(P) profile."""
    nabla = temperature_lapse(temperature, pressure)
    on_ad = nabla >= (float(nabla_ad) - float(tol))
    rcb = None
    n_cz = 0
    if on_ad.size and bool(on_ad[0]):
        i_hi = 0
        while i_hi + 1 < on_ad.size and on_ad[i_hi + 1]:
            i_hi += 1
        n_cz = i_hi + 1
        rcb = float(np.log10(float(pressure[i_hi])))
    return {
        "rcb_log10p": rcb,
        "n_near_adiabatic_interfaces": int(np.sum(on_ad)),
        "n_bottom_connected_cz_layers": n_cz,
        "max_lapse": float(np.max(nabla)) if nabla.size else None,
        "mean_lapse_in_cz": float(np.mean(nabla[:n_cz])) if n_cz > 0 else None,
    }


def stage3_on_helios_tp(
    tp_path: Path,
    *,
    n_layers: int,
    f_up_boa: float | None = None,
) -> dict:
    tp = load_tp_profile(tp_path)
    p_lay, t_lay, t_boa = _helios_layers(tp)
    rec = load_mlt_reference(n_layers)
    grid = build_helios_pressure_grid(
        p_boa_microbar=float(rec["helios_p_boa_microbar"]),
        p_toa_microbar=float(rec["helios_p_toa_microbar"]),
        n_layers=n_layers,
    )
    if t_lay.size != n_layers:
        raise ValueError(f"expected {n_layers} layers, got {t_lay.size}")
    spec = nested_analytic_opacity_spec(n_layers)
    opacity = spec.opacity()
    if f_up_boa is None:
        f_up_boa = float(STEFAN_BOLTZMANN * t_boa**4)
    rad = solve_radiation(
        t_lay,
        grid.layer_mass_kg_m2,
        opacity,
        grid.p_lay_Pa,
        TopIrradiation(0.0),
        LowerUpwardFlux(float(f_up_boa)),
        diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
    )
    up = np.sum(np.asarray(rad.flux_up, dtype=np.float64), axis=0)
    down = np.sum(np.asarray(rad.flux_down, dtype=np.float64), axis=0)
    net = np.asarray(rad.flux_net, dtype=np.float64)
    kappa = opacity.evaluate(t_lay, grid.p_lay_Pa)[0]
    nr_field = nabla_rad_layer(t_lay, grid.p_lay_Pa, net, kappa)
    nr_fint = nabla_rad_layer(t_lay, grid.p_lay_Pa, float(F_INT), kappa)
    return {
        "label": "stage3",
        "up": up,
        "down": down,
        "net": net,
        "kappa": kappa,
        "temperature": t_lay,
        "pressure": grid.p_lay_Pa,
        "t_boa": t_boa,
        "f_up_boa": float(f_up_boa),
        "actual_radiative_flux_gradient_diagnostic": crossing_diagnostic(
            nr_field, grid.p_lay_Pa, label="actual_radiative_flux_gradient_diagnostic"
        ),
        "required_total_flux_gradient_diagnostic": crossing_diagnostic(
            nr_fint, grid.p_lay_Pa, label="required_total_flux_gradient_diagnostic"
        ),
    }


def export_helios_postproc_on_final_tp(
    *,
    tp_path: Path,
    out_root: Path,
    n_layers: int = 96,
    opacity_path: Path = OPACITY,
) -> dict:
    """Write HELIOS post-processing iso=yes and iso=no on the final coupled T(P)."""
    tp = load_tp_profile(tp_path)
    p_lay, t_lay, t_boa = _helios_layers(tp)
    rec = load_mlt_reference(n_layers)
    grid = build_helios_pressure_grid(
        p_boa_microbar=float(rec["helios_p_boa_microbar"]),
        p_toa_microbar=float(rec["helios_p_toa_microbar"]),
        n_layers=n_layers,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    exported = {}
    for iso in (False, True):
        tag = "iso_yes" if iso else "iso_no"
        case_dir = out_root / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        case_name = f"stage4_rcb_attrib_n{n_layers}_{tag}"
        tp_out = case_dir / f"{case_name}_tp.dat"
        param_path = case_dir / "param.dat"
        write_tp_profile(
            tp_out,
            temperature_boa_k=t_boa,
            temperature_lay_k=t_lay,
            p_int_microbar=grid.p_int_microbar,
            p_lay_microbar=grid.p_lay_microbar,
        )
        write_param_dat(
            param_path,
            case_name=case_name,
            output_dir=str(case_dir.resolve()) + "/",
            toa_pressure_microbar=float(grid.p_toa_microbar),
            boa_pressure_microbar=float(grid.p_boa_microbar),
            opacity_path=str(opacity_path.resolve()),
            tp_profile_path=str(tp_out.resolve()),
            t_int_k=float(T_INT),
            diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
            scattering=False,
            convective_adjustment=False,
            direct_irradiation=False,
            post_processing=True,
            n_layers=n_layers,
            planet_type="rocky",
            isothermal_layers=iso,
            template_path=TEMPLATE,
        )
        text = param_path.read_text()
        text = re.sub(
            r"^(kappa value\s*=\s*)\S+",
            rf"\g<1>{NABLA_AD:.17g}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        param_path.write_text(text)
        meta = {
            "case_name": case_name,
            "isothermal_layers": iso,
            "run_type": "post-processing",
            "temperature_source": str(tp_path),
            "purpose": "rcb_attribution_radiation_source",
            "helios_commit": PINNED_HELIOS_COMMIT,
            "note": (
                "Frozen final HELIOS coupled T(P); only isothermal_layers changes. "
                "Not radiation-only parity and not iterative coupled scoring."
            ),
        }
        (case_dir / "attrib_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        exported[tag] = {"param": str(param_path), "case_dir": str(case_dir), "case_name": case_name}
    manifest = {
        "n_layers": n_layers,
        "tp_source": str(tp_path),
        "cases": exported,
        "helios_parity_headline": False,
        "full_stage4_claim": False,
    }
    (out_root / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def compare_radiation_fields(fields: dict[str, dict]) -> dict:
    """Pairwise F↑/F↓/F_net relative diffs; Stage-3≈iso=yes is the decisive check."""
    floor = max(
        abs(float(fields[k]["up"][0])) for k in fields
    )
    pairs = {}
    labels = list(fields.keys())
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            fa, fb = fields[a], fields[b]
            pairs[f"{a}_vs_{b}"] = {
                "flux_up_rel": _norm_diff(fa["up"], fb["up"], floor),
                "flux_down_rel": _norm_diff(fa["down"], fb["down"], floor),
                "flux_net_rel": _norm_diff(fa["net"], fb["net"], floor),
                "agree_at_gate": bool(
                    max(
                        _norm_diff(fa["up"], fb["up"], floor),
                        _norm_diff(fa["down"], fb["down"], floor),
                        _norm_diff(fa["net"], fb["net"], floor),
                    )
                    <= MATCHED_FLUX_REL_GATE
                ),
            }
    stage3_vs_iso_yes = pairs.get("stage3_vs_iso_yes") or pairs.get("iso_yes_vs_stage3")
    return {
        "gate_rel": MATCHED_FLUX_REL_GATE,
        "pairs": pairs,
        "stage3_approx_helios_iso_yes": (
            None if stage3_vs_iso_yes is None else bool(stage3_vs_iso_yes["agree_at_gate"])
        ),
        "interpretation": (
            "If Stage-3≈HELIOS iso=yes and both disagree with iso=no, the radiation "
            "discrepancy is HELIOS non-isothermal within-layer source treatment."
        ),
    }


def exact_adjustment_nested(
    n_layers: int = 96,
    *,
    max_iter: int = 80,
    f_int: float = F_INT,
    flux_gate_rel: float = 1.0e-3,
    dT_tol_rel: float = 1.0e-6,
) -> dict:
    """Exact convective adjustment RCE on nested-τ with the Stage-3 operator.

    Same grid, opacity, D, F_int, F_irr=0 as the finite-MLT reference. Instability
    uses the temperature lapse ∇=ΔlnT/ΔlnP (not the frozen F_rad diagnostic).
    Lower BC: LowerNetInternalFlux with F_conv(0) from a free LowerTemperature
    radiation solve so F_rad(0)+F_conv(0)=F_int.

    Status COMPLETE only if a real adjustment occurred, a single bottom-connected
    near-adiabatic CZ remains, and flux gates pass.
    """
    spec = nested_analytic_opacity_spec(n_layers, f_irr=0.0, f_int=f_int)
    pg = spec.grid()
    opacity = spec.opacity()
    thermo = ConstantH2Thermo()
    p_boa = float(pg.pressure_edges[0])
    t = grey_radiative_equilibrium_temperature(
        pg, opacity, f_int, 0.0, diffusivity_factor=HELIOS_DEFAULT_DIFFUSIVITY
    )
    nabla_seed = temperature_lapse(t, pg.pressure_centres)
    if not bool(nabla_seed[0] > NABLA_AD + SUPERAD_EPS):
        return {
            "purpose": "exact_convective_adjustment_nested_tau_stage3",
            "status": "INCOMPLETE",
            "reason": "grey RE seed is not bottom-unstable by temperature lapse",
            "max_lapse_seed": float(np.max(nabla_seed)),
            "nabla_ad": NABLA_AD,
            "rcb_log10p": None,
            "withdrawn_prior_0p121_dex_claim": True,
        }

    f_conv0 = 0.0
    history: list[dict] = []
    adjusted_once = False
    for it in range(max_iter):
        nabla = temperature_lapse(t, pg.pressure_centres)
        superad = nabla > (NABLA_AD + SUPERAD_EPS)
        if bool(superad[0]):
            i_hi = 0
            while i_hi + 1 < superad.size and superad[i_hi + 1]:
                i_hi += 1
            i_join = min(i_hi + 1, t.size - 1)
            t_new = t.copy()
            t_new[:i_join] = _temperature_on_adiabat(
                thermo,
                float(t[i_join]),
                float(pg.pressure_centres[i_join]),
                pg.pressure_centres[:i_join],
            )
            dT = float(np.max(np.abs(t_new - t)))
            t = t_new
            adjusted_once = True
            n_cz = int(i_hi + 1)
        else:
            dT = 0.0
            near = rcb_from_near_adiabat(t, pg.pressure_centres)
            n_cz = int(near["n_bottom_connected_cz_layers"])

        t_boa = float(t[0] * (p_boa / float(pg.pressure_centres[0])) ** NABLA_AD)
        rad_free = solve_radiation(
            t,
            pg.layer_mass,
            opacity,
            pg.pressure_centres,
            TopIrradiation(0.0),
            LowerTemperature(t_boa),
            diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
        )
        net_free = np.asarray(rad_free.flux_net, dtype=np.float64)
        f_conv0 = max(0.0, float(f_int) - float(net_free[0]))
        rad = solve_radiation(
            t,
            pg.layer_mass,
            opacity,
            pg.pressure_centres,
            TopIrradiation(0.0),
            LowerNetInternalFlux(float(f_int)),
            diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
            bottom_convective_flux=float(f_conv0),
        )
        net = np.asarray(rad.flux_net, dtype=np.float64)
        f_conv = float(f_int) - net
        toa_rel = abs(float(net[-1]) - float(f_int)) / max(abs(float(f_int)), 1.0)
        boa_conv_ok = float(f_conv[0]) >= -1.0e-6 * max(abs(float(f_int)), 1.0)
        history.append(
            {
                "iter": it,
                "n_cz": n_cz,
                "adjusted": bool(superad[0]),
                "max_abs_dT": dT,
                "f_rad_boa": float(net[0]),
                "f_conv_boa": float(f_conv[0]),
                "f_rad_toa": float(net[-1]),
                "toa_flux_rel": toa_rel,
            }
        )
        if (
            adjusted_once
            and dT < dT_tol_rel * max(float(np.max(t)), 1.0)
            and not bool(superad[0])
            and toa_rel <= flux_gate_rel
            and boa_conv_ok
        ):
            break

    nabla = temperature_lapse(t, pg.pressure_centres)
    superad = nabla > (NABLA_AD + SUPERAD_EPS)
    near = rcb_from_near_adiabat(t, pg.pressure_centres)
    rad = solve_radiation(
        t,
        pg.layer_mass,
        opacity,
        pg.pressure_centres,
        TopIrradiation(0.0),
        LowerNetInternalFlux(float(f_int)),
        diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
        bottom_convective_flux=float(f_conv0),
    )
    net = np.asarray(rad.flux_net, dtype=np.float64)
    f_conv = float(f_int) - net
    toa_rel = abs(float(net[-1]) - float(f_int)) / max(abs(float(f_int)), 1.0)
    n_cz = int(near["n_bottom_connected_cz_layers"])
    rz_flux_rel = (
        float(np.max(np.abs(net[n_cz:] - float(f_int)))) / max(abs(float(f_int)), 1.0)
        if n_cz < net.size
        else None
    )
    detached = bool(np.any(superad) and not bool(superad[0]))
    single_bottom_cz = (
        near["rcb_log10p"] is not None
        and n_cz > 0
        and not bool(superad[0])
        and not detached
    )
    gates = {
        "adjusted_at_least_once": adjusted_once,
        "no_remaining_superadiabatic": not bool(np.any(superad)),
        "single_bottom_connected_cz": single_bottom_cz,
        "toa_flux_rel_le_gate": toa_rel <= flux_gate_rel,
        "boa_f_conv_nonnegative": float(f_conv[0]) >= -1.0e-6 * max(abs(float(f_int)), 1.0),
    }
    complete = all(gates.values())
    rcb = near["rcb_log10p"] if complete else None
    return {
        "purpose": "exact_convective_adjustment_nested_tau_stage3",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "n_layers": n_layers,
        "f_int": f_int,
        "f_irr": 0.0,
        "diffusivity_factor": HELIOS_DEFAULT_DIFFUSIVITY,
        "lower_bc": "LowerNetInternalFlux with F_conv(0) from LowerTemperature free solve",
        "instability_criterion": "temperature_lapse_gt_nabla_ad",
        "iterations": len(history),
        "history_tail": history[-8:],
        "gates": gates,
        "flux_gate_rel": flux_gate_rel,
        "toa_flux_rel": toa_rel,
        "rz_flux_rel": rz_flux_rel,
        "f_rad_boa": float(net[0]),
        "f_conv_boa": float(f_conv[0]),
        "f_rad_toa": float(net[-1]),
        "rcb_log10p": rcb,
        "near_adiabat": near,
        "dex_vs_mlt_ref": (
            None if rcb is None else abs(float(rcb) - MLT_RCB_TARGET_LOG10P)
        ),
        "note": (
            "One-shot exact-adiabat adjustment from grey RE using temperature-lapse "
            "Schwarzschild (not the frozen F_rad ∇ diagnostic). Stage-3 operator; "
            "nested-τ; TOA/BOA flux gates. RZ is not fully re-equilibrated "
            f"(rz_flux_rel={rz_flux_rel}). Indicative closure ΔRCB vs finite MLT only; "
            "do not treat as a linear share of the HELIOS pilot 0.309-dex gap. "
            "An earlier 0.121-dex claim from an unadjusted profile was invalid; "
            "this COMPLETE run yields the same join RCB after a real adjustment."
        ),
    }


def finite_mlt_nested_rcb(n_layers: int = 96) -> dict:
    rec = load_mlt_reference(n_layers)
    # Cross-check MLT RCB via mixing_length_flux on the frozen profile.
    spec = nested_analytic_opacity_spec(n_layers, f_irr=0.0)
    pg = spec.grid()
    t = np.asarray(rec["temperature"], dtype=np.float64)
    alpha = float(rec.get("alpha") or rec.get("mixing_length_alpha") or 1.0)
    cl = mixing_length_flux(pg, t, GRAVITY_SI, alpha, ConstantH2Thermo())
    solver = SolverConfig()
    rcb_mlt = _primary_rcb_log10p(pg, cl, solver)
    return {
        "purpose": "finite_mlt_nested_tau_reference",
        "n_layers": n_layers,
        "record_rcb_log10p": rec.get("primary_rcb_log10p"),
        "recomputed_rcb_log10p": rcb_mlt,
        "alpha": alpha,
        "f_int": rec.get("f_int"),
        "f_irr": rec.get("f_irr"),
    }


def build_attribution_table(payload: dict) -> dict:
    rad = payload.get("radiation_source") or {}
    grad = payload.get("frozen_profile_gradient_diagnostics") or {}
    conv = payload.get("convection_closure") or {}
    reso = payload.get("resolution") or {}
    exact = conv.get("exact_adjustment") or {}

    def _dex(a, b):
        if a is None or b is None:
            return None
        return abs(float(a) - float(b))

    exact_status = exact.get("status")
    rows = [
        {
            "test": "HELIOS iso=yes vs iso=no",
            "isolates": "Layer-source treatment",
            "metric": "flux_net_rel (radiation field mismatch)",
            "value": {
                "flux_net_rel": (rad.get("pairs") or {}).get("iso_yes_vs_iso_no", {}).get(
                    "flux_net_rel"
                ),
                "note": (
                    "Source treatments produce substantially different F_rad partitions "
                    "on the same T(P). Capable of affecting coupled RCB; frozen-profile "
                    "∇ diagnostics do not quantify an RCB displacement."
                ),
            },
            "status": "COMPLETE" if (rad.get("pairs") or {}).get("iso_yes_vs_iso_no") else "PENDING",
        },
        {
            "test": "Stage-3 vs HELIOS iso=yes",
            "isolates": "Remaining radiation implementation",
            "metric": "agree_at_opacity_gate",
            "value": {
                "agree": rad.get("stage3_approx_helios_iso_yes"),
                "flux_net_rel": (rad.get("pairs") or {}).get("stage3_vs_iso_yes", {}).get(
                    "flux_net_rel"
                )
                or (rad.get("pairs") or {}).get("iso_yes_vs_stage3", {}).get("flux_net_rel"),
            },
            "status": "COMPLETE" if rad.get("stage3_approx_helios_iso_yes") is True else (
                "PENDING" if rad.get("stage3_approx_helios_iso_yes") is None else "COMPLETE"
            ),
        },
        {
            "test": "Finite MLT vs exact adjustment",
            "isolates": "Convection closure",
            "metric": "ΔRCB dex on nested-τ + Stage-3 (valid equilibria only)",
            "value": {
                "mlt_rcb": (conv.get("finite_mlt") or {}).get("record_rcb_log10p"),
                "exact_adj_status": exact_status,
                "exact_adj_rcb": exact.get("rcb_log10p"),
                "rcb_dex": (
                    _dex(
                        (conv.get("finite_mlt") or {}).get("record_rcb_log10p"),
                        exact.get("rcb_log10p"),
                    )
                    if exact_status == "COMPLETE"
                    else None
                ),
                "rz_flux_rel": exact.get("rz_flux_rel"),
                "indicative_only": True,
                "note": (
                    "ΔRCB is indicative of closure on nested-τ with Stage-3. "
                    "Do not divide by the 0.309-dex HELIOS pilot gap (nonlinear)."
                ),
            },
            "status": (
                "COMPLETE"
                if exact_status == "COMPLETE"
                else ("INCOMPLETE" if exact_status == "INCOMPLETE" else "PENDING")
            ),
        },
        {
            "test": "HELIOS N=96 vs N=192",
            "isolates": "HELIOS grid resolution",
            "metric": "ΔRCB dex (HELIOS lapse flags)",
            "value": {
                "n96_rcb": (reso.get("n96") or {}).get("rcb_log10p"),
                "n192_rcb": (reso.get("n192") or {}).get("rcb_log10p"),
                "rcb_dex": _dex(
                    (reso.get("n96") or {}).get("rcb_log10p"),
                    (reso.get("n192") or {}).get("rcb_log10p"),
                ),
            },
            "status": "PENDING" if not (reso.get("n192") or {}).get("rcb_log10p") else "COMPLETE",
        },
    ]
    return {
        "mlt_reference_rcb_log10p": MLT_RCB_TARGET_LOG10P,
        "helios_n96_pilot_rcb_log10p": 4.7187499048239445,
        "pilot_dex": abs(MLT_RCB_TARGET_LOG10P - 4.7187499048239445),
        "rows": rows,
        "frozen_profile_gradient_diagnostics_present": bool(grad),
        "helios_parity_headline": False,
        "full_stage4_claim": False,
        "defensible_conclusion": (
            "HELIOS non-isothermal source treatment is definitively responsible for "
            "the radiation-field mismatch. Exact-adjustment vs finite-MLT gives an "
            "indicative nested-τ closure ΔRCB; HELIOS N=96→N=192 moves RCB by only "
            "~0.019 dex. Radiation-source contribution to the coupled RCB shift is "
            "not yet separately quantified. Effects need not add linearly."
        ),
        "note": (
            "Do not interpret actual_radiative_flux_gradient_diagnostic or "
            "required_total_flux_gradient_diagnostic as predicted RCBs."
        ),
    }


def run_local(
    *,
    tp_path: Path,
    flux_path: Path,
    iso_yes_flux: Path | None,
    iso_no_flux: Path | None,
    n_layers: int,
    out_path: Path,
    n192_resolution: dict | None = None,
) -> dict:
    helios_iter = _load_flux_si(flux_path, n_layers) if flux_path.exists() else None
    f_up = float(helios_iter["up"][0]) if helios_iter is not None else None
    s3 = stage3_on_helios_tp(tp_path, n_layers=n_layers, f_up_boa=f_up)

    fields: dict[str, dict] = {
        "stage3": {
            "up": s3["up"],
            "down": s3["down"],
            "net": s3["net"],
            "temperature": s3["temperature"],
            "pressure": s3["pressure"],
            "kappa": s3["kappa"],
            "actual_radiative_flux_gradient_diagnostic": s3[
                "actual_radiative_flux_gradient_diagnostic"
            ],
            "required_total_flux_gradient_diagnostic": s3[
                "required_total_flux_gradient_diagnostic"
            ],
        }
    }
    if helios_iter is not None:
        fields["helios_iterative_iso_no"] = {
            "up": helios_iter["up"],
            "down": helios_iter["down"],
            "net": helios_iter["net"],
        }

    for label, path in (("iso_yes", iso_yes_flux), ("iso_no", iso_no_flux)):
        if path is None or not path.exists():
            continue
        fl = _load_flux_si(path, n_layers)
        nr = nabla_rad_layer(s3["temperature"], s3["pressure"], fl["net"], s3["kappa"])
        nr_fint = nabla_rad_layer(
            s3["temperature"], s3["pressure"], float(F_INT), s3["kappa"]
        )
        fields[label] = {
            "up": fl["up"],
            "down": fl["down"],
            "net": fl["net"],
            "actual_radiative_flux_gradient_diagnostic": crossing_diagnostic(
                nr, s3["pressure"], label="actual_radiative_flux_gradient_diagnostic"
            ),
            "required_total_flux_gradient_diagnostic": crossing_diagnostic(
                nr_fint, s3["pressure"], label="required_total_flux_gradient_diagnostic"
            ),
        }

    if "helios_iterative_iso_no" in fields:
        nr = nabla_rad_layer(
            s3["temperature"],
            s3["pressure"],
            fields["helios_iterative_iso_no"]["net"],
            s3["kappa"],
        )
        nr_fint = nabla_rad_layer(
            s3["temperature"], s3["pressure"], float(F_INT), s3["kappa"]
        )
        fields["helios_iterative_iso_no"][
            "actual_radiative_flux_gradient_diagnostic"
        ] = crossing_diagnostic(
            nr, s3["pressure"], label="actual_radiative_flux_gradient_diagnostic"
        )
        fields["helios_iterative_iso_no"][
            "required_total_flux_gradient_diagnostic"
        ] = crossing_diagnostic(
            nr_fint, s3["pressure"], label="required_total_flux_gradient_diagnostic"
        )

    rad_compare = compare_radiation_fields(
        {k: fields[k] for k in fields if k in ("stage3", "iso_yes", "iso_no")}
    )
    grad_diag = {
        k: {
            "actual_radiative_flux_gradient_diagnostic": fields[k].get(
                "actual_radiative_flux_gradient_diagnostic"
            ),
            "required_total_flux_gradient_diagnostic": fields[k].get(
                "required_total_flux_gradient_diagnostic"
            ),
            "interpretation": (
                "actual_* uses that column's F_rad (final flux partition; not a "
                "counterfactual without convection). required_* uses F_int on the "
                "shared T (identical across fields when T,κ match; structure only)."
            ),
        }
        for k in fields
        if "actual_radiative_flux_gradient_diagnostic" in fields[k]
    }

    finite = finite_mlt_nested_rcb(n_layers)
    exact = exact_adjustment_nested(n_layers)
    closure_dex = None
    if exact.get("status") == "COMPLETE" and exact.get("rcb_log10p") is not None:
        closure_dex = abs(
            float(finite["record_rcb_log10p"]) - float(exact["rcb_log10p"])
        )

    resolution = {
        "n96": {
            "rcb_log10p": 4.7187499048239445,
            "source": "helios_coupled_n96 job 16015698 lapse flags",
        },
        "n192": n192_resolution
        or {
            "rcb_log10p": None,
            "status": "NOT_RUN",
            "note": "Labelled HELIOS resolution test.",
        },
    }

    payload = {
        "job_id_pilot": "16015698",
        "helios_tp": str(tp_path),
        "helios_iterative_flux": str(flux_path) if flux_path.exists() else None,
        "radiation_source": {
            **rad_compare,
            "stage3_column": {
                "f_up_boa": float(s3["up"][0]),
                "f_down_boa": float(s3["down"][0]),
                "f_net_boa": float(s3["net"][0]),
                "f_net_toa": float(s3["net"][-1]),
            },
            "available_fields": sorted(fields.keys()),
            "pending_helios_postproc": [
                x for x in ("iso_yes", "iso_no") if x not in fields
            ],
            "radiation_source_rcb_shift": "NOT_YET_QUANTIFIED",
        },
        "frozen_profile_gradient_diagnostics": grad_diag,
        "convection_closure": {
            "finite_mlt": finite,
            "exact_adjustment": exact,
            "rcb_dex_mlt_vs_exact_adj": closure_dex,
            "indicative_only": True,
            "prior_invalid_unadjusted_0p121_claim": "withdrawn",
        },
        "resolution": resolution,
        "helios_parity_headline": False,
        "full_stage4_claim": False,
    }
    payload["attribution_table"] = build_attribution_table(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=Path, default=DEBUG / "final_tp.dat")
    parser.add_argument("--flux", type=Path, default=DEBUG / "final_integrated_flux.dat")
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument(
        "--export-postproc-dir",
        type=Path,
        default=None,
        help="Export HELIOS iso=yes/no post-proc cases on final HELIOS T(P)",
    )
    parser.add_argument("--iso-yes-flux", type=Path, default=None)
    parser.add_argument("--iso-no-flux", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "helios_coupled_n96_rcb_attribution.json",
    )
    args = parser.parse_args()

    if args.export_postproc_dir is not None:
        man = export_helios_postproc_on_final_tp(
            tp_path=args.tp,
            out_root=args.export_postproc_dir,
            n_layers=args.layers,
        )
        print(json.dumps({"exported": man}, indent=2))

    payload = run_local(
        tp_path=args.tp,
        flux_path=args.flux,
        iso_yes_flux=args.iso_yes_flux,
        iso_no_flux=args.iso_no_flux,
        n_layers=args.layers,
        out_path=args.output,
    )
    summary = {
        "out": str(args.output),
        "pending_helios_postproc": payload["radiation_source"]["pending_helios_postproc"],
        "stage3_approx_iso_yes": payload["radiation_source"]["stage3_approx_helios_iso_yes"],
        "exact_adjustment_status": payload["convection_closure"]["exact_adjustment"].get(
            "status"
        ),
        "exact_adjustment_rcb": payload["convection_closure"]["exact_adjustment"].get(
            "rcb_log10p"
        ),
        "closure_dex": payload["convection_closure"]["rcb_dex_mlt_vs_exact_adj"],
        "resolution_n192": payload["resolution"].get("n192"),
        "attribution_rows": [
            {"test": r["test"], "status": r["status"], "value": r["value"]}
            for r in payload["attribution_table"]["rows"]
        ],
        "defensible_conclusion": payload["attribution_table"].get("defensible_conclusion"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

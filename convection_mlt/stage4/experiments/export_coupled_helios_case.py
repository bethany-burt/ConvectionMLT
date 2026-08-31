"""Export coupled HELIOS case + cheap frozen-T iso=no radiation diagnostic.

The iterative coupled param uses isothermal layers = no (required for HELIOS
convective adjustment at b0800f9). A sibling post-processing run on the same
MLT F_irr=0 T(P) quantifies the iso=yes → iso=no flux change before iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt.adapters.helios import write_param_dat, write_tp_profile
from convection_mlt.adapters.helios_contracts import (
    F_INT,
    GRAVITY_SI,
    HELIOS_DEFAULT_DIFFUSIVITY,
    PINNED_HELIOS_COMMIT,
    PROVENANCE_ONLY,
    T_INT,
)
from convection_mlt.adapters.helios_grid import build_helios_pressure_grid
from compare_coupled_helios_rce import interpolate_temperature, load_mlt_reference

FIXTURES = ROOT / "fixtures" / "helios"
OPACITY = FIXTURES / "analytic_grey_nested.h5"
TEMPLATE = FIXTURES / "helios_param_template.dat"
TOLERANCES = FIXTURES / "coupled_rce_benchmark_tolerances.json"
MANIFEST = FIXTURES / "coupled_input_manifest.json"
NABLA_AD = float(PROVENANCE_ONLY["nabla_ad"])
# Pinned HELIOS Planck table — do not enlarge for numerical runaway.
PLANCK_TABLE_DIM = 8000
PLANCK_TABLE_STEP = 2
PLANCK_T_CEILING_K = PLANCK_TABLE_DIM * PLANCK_TABLE_STEP - 2  # 15998 K


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_manifest(name: str, path: Path) -> None:
    man = json.loads(MANIFEST.read_text())
    expected = (man.get("files") or {}).get(name)
    if expected is None:
        raise SystemExit(f"{name} is not pinned in {MANIFEST}")
    if not path.exists():
        raise SystemExit(f"missing pinned file {path}")
    observed = _sha256(path)
    if observed != expected:
        raise SystemExit(
            f"{name} sha256={observed} != pinned {expected}. Refusing to export."
        )


def _patch_kappa(param_path: Path) -> None:
    text = param_path.read_text()
    text = re.sub(
        r"^(kappa value\s*=\s*)\S+",
        rf"\g<1>{NABLA_AD:.17g}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    param_path.write_text(text)


def _assert_planck_table_unchanged(param_text: str) -> None:
    m = re.search(
        r"^plancktable dimension and stepsize\s*=\s*(\S+)\s+(\S+)",
        param_text,
        re.M,
    )
    if not m:
        raise SystemExit("missing plancktable dimension and stepsize in param.dat")
    dim, step = int(float(m.group(1))), int(float(m.group(2)))
    if dim != PLANCK_TABLE_DIM or step != PLANCK_TABLE_STEP:
        raise SystemExit(
            f"Planck table was changed to {dim} {step}; refusing. "
            f"Pinned {PLANCK_TABLE_DIM} {PLANCK_TABLE_STEP}. "
            "Do not enlarge the table for numerical runaway."
        )


def _mlt_on_helios_grid(n_layers: int, opacity_path: Path):
    _require_manifest("coupled_rce_benchmark_tolerances.json", TOLERANCES)
    mlt_name = f"mlt_nested_tau_n{n_layers}_firr0.json"
    mlt_path = FIXTURES / mlt_name
    seed_note = None
    if mlt_path.exists():
        if mlt_name in (json.loads(MANIFEST.read_text()).get("files") or {}):
            _require_manifest(mlt_name, mlt_path)
        rec = load_mlt_reference(n_layers)
    elif n_layers == 192 and (FIXTURES / "mlt_nested_tau_n96_firr0.json").exists():
        # Labelled HELIOS resolution test: sample N=96 F_irr=0 MLT onto N=192 HELIOS grid.
        rec = load_mlt_reference(96)
        seed_note = (
            "N=192 resolution test seeds from mlt_nested_tau_n96_firr0.json sampled "
            "onto the N=192 HELIOS geometric grid. Not a headline N=192 MLT reference."
        )
    else:
        raise SystemExit(f"missing F_irr=0 MLT reference {mlt_path}")
    if "f_irr" not in rec or float(rec["f_irr"]) != 0.0:
        raise SystemExit("MLT reference f_irr must be 0")
    p_boa = float(rec["helios_p_boa_microbar"])
    p_toa = float(rec["helios_p_toa_microbar"])
    grid = build_helios_pressure_grid(
        p_boa_microbar=p_boa, p_toa_microbar=p_toa, n_layers=n_layers
    )
    t_lay = interpolate_temperature(
        np.log(np.asarray(rec["pressure_centres"], dtype=np.float64)),
        rec["temperature"],
        np.log(grid.p_lay_Pa),
    )
    t_boa = float(
        interpolate_temperature(
            np.log(np.asarray(rec["pressure_centres"], dtype=np.float64)),
            rec["temperature"],
            np.log(np.array([grid.p_int_Pa[0]])),
        )[0]
    )
    if float(np.max(t_lay)) >= 0.5 * PLANCK_T_CEILING_K or t_boa >= 0.5 * PLANCK_T_CEILING_K:
        raise SystemExit(
            f"MLT seed T approaches Planck ceiling ({PLANCK_T_CEILING_K} K); "
            "refusing export (physical reference should be well below 700 K)."
        )
    if seed_note is not None:
        rec = dict(rec)
        rec["resolution_test_seed_note"] = seed_note
    return rec, grid, t_lay, t_boa, opacity_path


def export_frozen_iso_diag(
    n_layers: int,
    *,
    case_dir: Path,
    opacity_path: Path,
    isothermal_layers: bool,
) -> Path:
    """Write post-processing param+TP for frozen-T radiation diagnostic."""
    rec, grid, t_lay, t_boa, opacity_path = _mlt_on_helios_grid(n_layers, opacity_path)
    case_dir.mkdir(parents=True, exist_ok=True)
    tag = "iso_yes" if isothermal_layers else "iso_no"
    case_name = f"stage4_coupled_n{n_layers}_frozen_{tag}"
    tp_path = case_dir / f"{case_name}_tp.dat"
    param_path = case_dir / "param.dat"
    write_tp_profile(
        tp_path,
        temperature_boa_k=t_boa,
        temperature_lay_k=t_lay,
        p_int_microbar=grid.p_int_microbar,
        p_lay_microbar=grid.p_lay_microbar,
    )
    write_param_dat(
        param_path,
        case_name=case_name,
        output_dir=str(case_dir) + "/",
        toa_pressure_microbar=float(grid.p_toa_microbar),
        boa_pressure_microbar=float(grid.p_boa_microbar),
        opacity_path=str(opacity_path),
        tp_profile_path=str(tp_path),
        t_int_k=float(T_INT),
        diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
        scattering=False,
        convective_adjustment=False,
        direct_irradiation=False,
        post_processing=True,
        n_layers=n_layers,
        planet_type="rocky",
        isothermal_layers=isothermal_layers,
        template_path=TEMPLATE,
    )
    _patch_kappa(param_path)
    _assert_planck_table_unchanged(param_path.read_text())
    meta = {
        "case_name": case_name,
        "isothermal_layers": bool(isothermal_layers),
        "run_type": "post-processing",
        "convective_adjustment": False,
        "mlt_profile_checksum_sha256": rec.get("profile_checksum_sha256"),
        "t_boa_k": t_boa,
        "t_lay_max_k": float(np.max(t_lay)),
        "t_lay_min_k": float(np.min(t_lay)),
        "planck_table": [PLANCK_TABLE_DIM, PLANCK_TABLE_STEP],
        "planck_t_ceiling_k": PLANCK_T_CEILING_K,
        "note": (
            "Frozen-T radiation diagnostic on MLT F_irr=0 seed. Not radiation-only "
            "parity and not the coupled iterative benchmark."
        ),
    }
    (case_dir / "frozen_diag_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return param_path


def export_coupled_case(
    n_layers: int = 96,
    *,
    case_dir: Path,
    opacity_path: Path = OPACITY,
    isothermal_layers: bool = False,
    labelled_iso1_counterfactual: bool = False,
) -> dict:
    rec, grid, t_lay, t_boa, opacity_path = _mlt_on_helios_grid(n_layers, opacity_path)
    case_dir.mkdir(parents=True, exist_ok=True)
    if labelled_iso1_counterfactual:
        case_name = f"stage4_coupled_n{n_layers}_iso_yes_adj"
    else:
        case_name = f"stage4_coupled_n{n_layers}"
    tp_path = case_dir / f"{case_name}_tp.dat"
    param_path = case_dir / "param.dat"
    write_tp_profile(
        tp_path,
        temperature_boa_k=t_boa,
        temperature_lay_k=t_lay,
        p_int_microbar=grid.p_int_microbar,
        p_lay_microbar=grid.p_lay_microbar,
    )
    write_param_dat(
        param_path,
        case_name=case_name,
        output_dir=str(case_dir) + "/",
        toa_pressure_microbar=float(grid.p_toa_microbar),
        boa_pressure_microbar=float(grid.p_boa_microbar),
        opacity_path=str(opacity_path),
        tp_profile_path=str(tp_path),
        t_int_k=float(T_INT),
        diffusivity_factor=float(HELIOS_DEFAULT_DIFFUSIVITY),
        scattering=False,
        convective_adjustment=True,
        direct_irradiation=False,
        post_processing=False,
        n_layers=n_layers,
        planet_type="rocky",
        isothermal_layers=bool(isothermal_layers),
        max_iterations=50000,
        template_path=TEMPLATE,
    )
    _patch_kappa(param_path)
    param_text = param_path.read_text()
    _assert_planck_table_unchanged(param_text)
    if labelled_iso1_counterfactual and not isothermal_layers:
        raise SystemExit("iso1 counterfactual requires isothermal_layers=True")
    runtime = {
        "purpose": (
            "Labelled N=96 HELIOS counterfactual: convective adjustment + iso=yes "
            "(minimal ISO1 conv_check patch). Isolates layer-source treatment vs "
            "iso=no pilot. Not Stage-4 headline; 0.15-dex RCB gate remains failed."
            if labelled_iso1_counterfactual
            else f"Declared N={n_layers} coupled-HELIOS runtime. Not a live HELIOS result."
        ),
        "comparison_type": "independently_discretized_rce_matched_forcing",
        "forcing": "F_int=300, F_irr=0",
        "f_irr": 0.0,
        "mlt_grid": rec.get("mlt_grid") or "nested_tau_interpolated_to_helios",
        "helios_commit": PINNED_HELIOS_COMMIT,
        "n_layers": n_layers,
        "run_type": "iterative",
        "physical_timestep": "no",
        "start_from_provided_tp_profile": "no",
        "isothermal_layers": bool(isothermal_layers),
        "maximum_number_of_iterations": 50000,
        "radiative_equilibrium_criterion": 1.0e-8,
        "relax_radiative_criterion_at": [10000, 20000],
        "initialization": (
            "stock_helios_isothermal_500K; iso=yes + conv adjust with versioned "
            "ISO1 patch (helios_iso1_conv_check_b0800f9.patch). Holds grid, "
            "adjustment, opacity, forcing, N=96 fixed; only layer-source changes."
            if labelled_iso1_counterfactual
            else (
                "stock_helios_isothermal_500K initial profile; isothermal_layers=false "
                "for non-iso radiation + convective adjustment (HELIOS b0800f9 requires "
                "iso==0 for conv_check). MLT tp.dat is provenance / later seed trial only."
            )
        ),
        "convective_adjustment": "yes",
        "kappa_nabla_ad": NABLA_AD,
        "scattering": "no",
        "direct_irradiation": "no",
        "diffusivity_factor": HELIOS_DEFAULT_DIFFUSIVITY,
        "internal_flux_temperature_k": float(T_INT),
        "f_int": F_INT,
        "gravity_si": GRAVITY_SI,
        "planet_type": "rocky",
        "precision": "double",
        "planck_table_dimension_and_stepsize": [PLANCK_TABLE_DIM, PLANCK_TABLE_STEP],
        "planck_t_ceiling_k": PLANCK_T_CEILING_K,
        "planck_table_policy": (
            "Do not enlarge. Physical reference T < 700 K; ceiling approach is "
            "numerical runaway — diagnose stability/damping instead."
        ),
        "prior_job_16015568": "HELIOS_RUNTIME_BUG_ISO1",
        "iso1_patch": (
            "helios_iso1_conv_check_b0800f9.patch" if labelled_iso1_counterfactual else None
        ),
        "labelled_iso1_source_counterfactual": bool(labelled_iso1_counterfactual),
        "opacity_path": str(opacity_path),
        "mlt_reference": str(FIXTURES / f"mlt_nested_tau_n{n_layers}_firr0.json"),
        "mlt_profile_checksum_sha256": rec.get("profile_checksum_sha256"),
        "source_status": rec.get("status"),
        "source_rcb_log10p": rec.get("primary_rcb_log10p"),
        "resolution_test_seed_note": rec.get("resolution_test_seed_note"),
        "labelled_resolution_test": n_layers == 192,
        "p_boa_microbar": grid.p_boa_microbar,
        "p_toa_microbar": grid.p_toa_microbar,
        "tolerances": str(TOLERANCES),
        "param_dat": str(param_path),
        "tp_seed": str(tp_path),
        "helios_coupled_rce_status": "NOT_RUN",
        "helios_parity_headline": False,
        "full_stage4_claim": False,
        "irradiated_nested_mlt": "structural_diagnostic_only",
    }
    runtime_path = case_dir / "helios_runtime_config.json"
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=96, choices=(96, 192))
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=ROOT / "results" / "helios_coupled_n96_case",
    )
    parser.add_argument("--opacity", type=Path, default=OPACITY)
    parser.add_argument(
        "--frozen-iso-diag-dir",
        type=Path,
        default=None,
        help="If set, also export frozen-T iso=no (+ iso=yes baseline) diagnostic cases.",
    )
    parser.add_argument(
        "--iso-yes-adj-counterfactual",
        action="store_true",
        help=(
            "Export labelled N=96 run with isothermal_layers=yes + convective "
            "adjustment (requires ISO1 HELIOS patch at runtime)."
        ),
    )
    args = parser.parse_args()
    payload = export_coupled_case(
        args.layers,
        case_dir=args.case_dir,
        opacity_path=args.opacity,
        isothermal_layers=bool(args.iso_yes_adj_counterfactual),
        labelled_iso1_counterfactual=bool(args.iso_yes_adj_counterfactual),
    )
    if args.frozen_iso_diag_dir is not None:
        base = args.frozen_iso_diag_dir
        export_frozen_iso_diag(
            args.layers,
            case_dir=base / "iso_no",
            opacity_path=args.opacity,
            isothermal_layers=False,
        )
        export_frozen_iso_diag(
            args.layers,
            case_dir=base / "iso_yes",
            opacity_path=args.opacity,
            isothermal_layers=True,
        )
        payload["frozen_iso_diag_dir"] = str(base)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()

"""Export nested MLT RCE as a HELIOS iterative + convective-adjustment case.

Does not run HELIOS. The N=96 coupled pilot uses the gated nested N=96
record as the T(P) seed, matched F_int / g / opacity / geometric grid, and
convective adjustment. HELIOS direct-beam irradiation does not map to
Stage-3 TopIrradiation (beam_contract.json); the pilot therefore uses no
stellar beam. Comparison to irradiated MLT is a benchmark, not identity.
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

from convection_mlt.adapters.helios import write_param_dat, write_tp_profile
from convection_mlt.adapters.helios_contracts import (
    F_INT,
    GRAVITY_SI,
    HELIOS_DEFAULT_DIFFUSIVITY,
    PINNED_HELIOS_COMMIT,
    PROVENANCE_ONLY,
    T_INT,
)
from convection_mlt.adapters.helios_grid import (
    build_helios_grid_from_nested_edges,
    sample_nested_tp_on_helios_grid,
)
from export_helios_grid_reference import _load_record

FIXTURES = ROOT / "fixtures" / "helios"
OPACITY = FIXTURES / "analytic_grey_nested.h5"
TEMPLATE = FIXTURES / "helios_param_template.dat"
TOLERANCES = FIXTURES / "coupled_rce_benchmark_tolerances.json"
NABLA_AD = float(PROVENANCE_ONLY["nabla_ad"])


def export_coupled_case(
    n_layers: int = 96,
    *,
    case_dir: Path,
    opacity_path: Path = OPACITY,
) -> dict:
    rec = _load_record(n_layers)
    edges = np.asarray(rec["pressure_edges"], dtype=np.float64)
    grid = build_helios_grid_from_nested_edges(edges, n_layers)
    temperature_boa_k, temperature_lay_k = sample_nested_tp_on_helios_grid(rec, grid)
    case_dir.mkdir(parents=True, exist_ok=True)
    case_name = f"stage4_coupled_n{n_layers}"
    tp_path = case_dir / f"{case_name}_tp.dat"
    param_path = case_dir / "param.dat"
    write_tp_profile(
        tp_path,
        temperature_boa_k=temperature_boa_k,
        temperature_lay_k=temperature_lay_k,
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
        template_path=TEMPLATE,
    )
    # HELIOS kappa is ∇_ad = (γ-1)/γ for an ideal gas.
    text = param_path.read_text()
    text = text.replace(
        "kappa value =                                         0.285714",
        f"kappa value =                                         {NABLA_AD:.6g}                        ",
    )
    param_path.write_text(text)
    runtime = {
        "purpose": "Declared N=96 coupled-HELIOS pilot runtime. Not a live HELIOS result.",
        "comparison_type": "benchmark_not_solver_identity",
        "helios_commit": PINNED_HELIOS_COMMIT,
        "n_layers": n_layers,
        "run_type": "iterative",
        "convective_adjustment": "yes",
        "kappa_nabla_ad": NABLA_AD,
        "scattering": "no",
        "direct_irradiation": "no",
        "irradiation_note": (
            "HELIOS direct beam does not map to TopIrradiation. "
            "Pilot uses F_int only; MLT nested record has F_irr=120 W m^-2."
        ),
        "diffusivity_factor": HELIOS_DEFAULT_DIFFUSIVITY,
        "internal_flux_temperature_k": T_INT,
        "f_int": F_INT,
        "gravity_si": GRAVITY_SI,
        "planet_type": "rocky",
        "precision": "double",
        "opacity_path": str(opacity_path),
        "source_record": (
            "n192_implicit_rce.json" if n_layers == 192
            else f"nested_rce_family.json[{n_layers}]"
        ),
        "source_profile_checksum_sha256": rec.get("profile_checksum_sha256")
        or rec.get("checksum_sha256"),
        "source_status": rec.get("status"),
        "source_rcb_log10p": rec.get("primary_rcb_log10p"),
        "p_boa_microbar": grid.p_boa_microbar,
        "p_toa_microbar": grid.p_toa_microbar,
        "tolerances": str(TOLERANCES),
        "param_dat": str(param_path),
        "tp_seed": str(tp_path),
    }
    runtime_path = case_dir / "helios_runtime_config.json"
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=ROOT / "results" / "helios_coupled_n96_case",
    )
    parser.add_argument("--opacity", type=Path, default=OPACITY)
    args = parser.parse_args()
    payload = export_coupled_case(
        args.layers, case_dir=args.case_dir, opacity_path=args.opacity
    )
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()

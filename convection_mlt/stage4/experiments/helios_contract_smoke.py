"""N=8 HELIOS contract smoke: grid, tp round-trip, orientation, rocky-surface BC gates."""

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

from convection_mlt import (
    LowerTemperature,
    STEFAN_BOLTZMANN,
    TopIrradiation,
    nested_analytic_opacity_spec,
    solve_radiation,
)
from convection_mlt.adapters.helios import (
    flux_cgs_to_si,
    load_integrated_flux,
    load_tp_profile,
    simulate_helios_tp_read,
    to_canonical_interfaces,
    write_param_dat,
    write_tp_profile,
)
from convection_mlt.adapters.helios_contracts import (
    F_INT,
    PINNED_HELIOS_COMMIT,
    T_INT,
    helios_track_status,
)
from convection_mlt.adapters.helios_grid import build_helios_pressure_grid
from convection_mlt.adapters.helios_opacity_table import build_table_arrays, write_helios_opacity_hdf5

RESULTS = ROOT / "results"
FIXTURES = ROOT / "fixtures" / "helios"

SMOKE_REL_TOL = 1.0e-3
SMOKE_ABS_TOL_W_M2 = 1.0
SMOKE_GRID_REL_TOL = 2.0e-6
SMOKE_DECOMP_REL_TOL = 1.0e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel_diff(observed: float, expected: float, *, floor: float = 1.0) -> float:
    scale = max(abs(expected), abs(observed), floor)
    return float(abs(observed - expected) / scale)


def _parse_param_runtime_config(param_path: Path) -> dict:
    text = param_path.read_text(encoding="utf-8")

    def _grab(key: str, cast=str):
        m = re.search(rf"^{re.escape(key)}\s+(.+?)\s+\[", text, flags=re.MULTILINE)
        if not m:
            return None
        val = m.group(1).strip()
        if cast is float:
            return float(val.split()[0])
        if cast is int:
            return int(float(val.split()[0]))
        return val

    return {
        "helios_commit": PINNED_HELIOS_COMMIT,
        "n_layers": _grab("number of layers =", int),
        "diffusivity_factor": _grab("diffusivity factor =", float),
        "precision": _grab("precision ="),
        "scattering": _grab("scattering ="),
        "convective_adjustment": _grab("convective adjustment ="),
        "direct_irradiation": _grab("direct irradiation beam ="),
        "planet_type": _grab("planet type ="),
        "surface_albedo": _grab("surface albedo =", float),
        "internal_flux_temperature_k": _grab("internal temperature [K] =", float),
        "p_boa_microbar": _grab("BOA pressure [10^-6 bar] =", float),
        "p_toa_microbar": _grab("TOA pressure [10^-6 bar] =", float),
        "run_type": _grab("run type ="),
        "stellar_model": _grab("stellar spectral model ="),
    }


def _manufactured_n8_profile(grid) -> tuple[float, np.ndarray]:
    t_boa = 640.0
    t_lay = 500.0 + 120.0 * np.sin(np.linspace(0.0, 1.8, grid.n_layers))
    return t_boa, t_lay


def prepare_smoke_case(case_dir: Path) -> dict:
    n = 8
    p_boa = 1.0e9
    p_toa = 10.0
    grid = build_helios_pressure_grid(p_boa_microbar=p_boa, p_toa_microbar=p_toa, n_layers=n)
    t_boa, t_lay = _manufactured_n8_profile(grid)
    case_dir.mkdir(parents=True, exist_ok=True)
    tp_path = case_dir / "smoke_n8_tp.dat"
    opacity_path = case_dir / "smoke_n8_opacity.h5"
    param_path = case_dir / "param.dat"
    write_tp_profile(
        tp_path,
        temperature_boa_k=t_boa,
        temperature_lay_k=t_lay,
        p_int_microbar=grid.p_int_microbar,
        p_lay_microbar=grid.p_lay_microbar,
    )
    spec = nested_analytic_opacity_spec(96)
    table = build_table_arrays(
        spec.opacity(),
        t_min=float(np.min(t_lay)) - 50.0,
        t_max=float(np.max([t_boa, t_lay.max()])) + 50.0,
        p_min_bar=float(grid.p_lay_microbar[-1]) * 1.0e-6,
        p_max_bar=float(grid.p_lay_microbar[0]) * 1.0e-6,
    )
    write_helios_opacity_hdf5(opacity_path, table)
    write_param_dat(
        param_path,
        case_name="helios_contract_smoke_n8",
        output_dir=f"{case_dir}/",
        toa_pressure_microbar=p_toa,
        boa_pressure_microbar=p_boa,
        opacity_path=f"{case_dir}/smoke_n8_opacity.h5",
        tp_profile_path=f"{case_dir}/smoke_n8_tp.dat",
        t_int_k=T_INT,
        diffusivity_factor=2.0,
        scattering=False,
        convective_adjustment=False,
        direct_irradiation=False,
        post_processing=True,
        n_layers=n,
        planet_type="rocky",
        surface_albedo=0.0,
    )
    return {
        "case_dir": str(case_dir),
        "tp_path": str(tp_path),
        "param_path": str(param_path),
        "opacity_path": str(opacity_path),
        "grid": {
            "p_int_microbar": grid.p_int_microbar.tolist(),
            "p_lay_microbar": grid.p_lay_microbar.tolist(),
            "layer_mass_kg_m2": grid.layer_mass_kg_m2.tolist(),
        },
        "temperature_boa_k": t_boa,
        "temperature_lay_k": t_lay.tolist(),
    }


def _stage3_rocky_reference(
    grid,
    *,
    temperature_boa_k: float,
    temperature_lay_k: np.ndarray,
) -> dict[str, np.ndarray]:
    spec = nested_analytic_opacity_spec(96)
    rad = solve_radiation(
        temperature_lay_k,
        grid.layer_mass_kg_m2,
        spec.opacity(),
        grid.p_lay_Pa,
        TopIrradiation(0.0),
        LowerTemperature(float(temperature_boa_k)),
        diffusivity_factor=2.0,
    )
    f_up = np.asarray(rad.flux_up.sum(axis=0), dtype=np.float64)
    f_down = np.asarray(rad.flux_down.sum(axis=0), dtype=np.float64)
    f_net = np.asarray(rad.flux_net, dtype=np.float64)
    return {"flux_up_W_m2": f_up, "flux_down_W_m2": f_down, "flux_net_W_m2": f_net}


def score_smoke(
    *,
    case_dir: Path,
    flux_path: Path | None,
    prep: dict,
    tolerances: dict | None = None,
) -> dict:
    n = 8
    grid = build_helios_pressure_grid(p_boa_microbar=1.0e9, p_toa_microbar=10.0, n_layers=n)
    tp_path = case_dir / "smoke_n8_tp.dat"
    param_path = case_dir / "param.dat"
    tp = load_tp_profile(tp_path)
    file_t = tp.temperature_k
    identity = simulate_helios_tp_read(tp.pressure_microbar, file_t, grid)
    expected = np.concatenate([[float(file_t[tp.layer_index == -1][0])], file_t[tp.layer_index >= 0]])
    t_boa = float(prep["temperature_boa_k"])
    runtime = _parse_param_runtime_config(param_path)
    sigma_tboa4 = float(STEFAN_BOLTZMANN * t_boa**4)

    contracts = {
        "helios_executable_smoke": {"status": "NOT_RUN"},
        "grid_contract": {"status": "NOT_RUN"},
        "tp_roundtrip_contract": {"status": "NOT_RUN"},
        "orientation_contract": {"status": "NOT_RUN"},
        "boundary_parameter_contract": {"status": "NOT_RUN"},
        "rocky_surface_radiative_bc": {"status": "NOT_RUN"},
        "flux_decomposition_contract": {"status": "NOT_RUN"},
        "stage3_rocky_surface_parity": {"status": "NOT_RUN"},
    }
    checks: dict[str, object] = {
        "tp_identity_max_rel": float(
            np.max(np.abs(identity - expected) / np.maximum(np.abs(expected), 1.0))
        ),
        "tp_target_pressures_exact": bool(
            np.allclose(tp.pressure_microbar, np.concatenate([[grid.p_int_microbar[0]], grid.p_lay_microbar]))
        ),
        "temperature_boa_k": t_boa,
        "sigma_Tboa4_W_m2": sigma_tboa4,
    }

    tp_ok = checks["tp_identity_max_rel"] <= SMOKE_REL_TOL and checks["tp_target_pressures_exact"]
    contracts["tp_roundtrip_contract"] = {
        "status": "PASS" if tp_ok else "FAIL",
        "checks": {
            "tp_identity_max_rel": checks["tp_identity_max_rel"],
            "tp_target_pressures_exact": checks["tp_target_pressures_exact"],
        },
    }

    status = "PENDING_LIVE"
    if flux_path and flux_path.exists():
        contracts["helios_executable_smoke"] = {"status": "PASS", "helios_flux_file": str(flux_path)}
        flux = load_integrated_flux(flux_path)
        n_int = flux.flux_net_cgs.size
        checks["n_interfaces_expected"] = n_int == n + 1

        net = to_canonical_interfaces(flux_cgs_to_si(flux.flux_net_cgs), flux.pressure_microbar, n_layers=n)
        up = to_canonical_interfaces(flux_cgs_to_si(flux.flux_up_cgs), flux.pressure_microbar, n_layers=n)
        down = to_canonical_interfaces(flux_cgs_to_si(flux.flux_down_cgs), flux.pressure_microbar, n_layers=n)
        intern = to_canonical_interfaces(flux_cgs_to_si(flux.flux_intern_cgs), flux.pressure_microbar, n_layers=n)

        checks["f_intern_boa_W_m2"] = float(intern[0])
        checks["f_up_boa_W_m2"] = float(up[0])
        checks["f_down_boa_W_m2"] = float(down[0])
        checks["f_net_boa_W_m2"] = float(net[0])
        checks["f_down_toa_W_m2"] = float(down[-1])
        checks["pressure_grid_max_rel"] = float(
            np.max(np.abs(flux.pressure_microbar - grid.p_int_microbar) / np.maximum(grid.p_int_microbar, 1.0))
        )
        decomp_rel = float(
            np.max(np.abs(net - (up - down)) / np.maximum.reduce([np.abs(net), np.abs(up - down), np.full(net.size, 1.0)]))
        )
        checks["flux_decomposition_max_rel"] = decomp_rel
        checks["rocky_surface_up_rel"] = _rel_diff(float(up[0]), sigma_tboa4, floor=sigma_tboa4)
        checks["rocky_surface_net_rel"] = _rel_diff(float(net[0]), sigma_tboa4 - float(down[0]), floor=sigma_tboa4)
        checks["f_intern_parameter_rel"] = _rel_diff(float(intern[0]), F_INT, floor=F_INT)

        grid_ok = checks["pressure_grid_max_rel"] <= SMOKE_GRID_REL_TOL and checks["n_interfaces_expected"]
        contracts["grid_contract"] = {
            "status": "PASS" if grid_ok else "FAIL",
            "checks": {
                "n_interfaces_expected": checks["n_interfaces_expected"],
                "pressure_grid_max_rel": checks["pressure_grid_max_rel"],
            },
        }
        contracts["orientation_contract"] = {
            "status": "PASS" if grid_ok else "FAIL",
            "note": "Bottom-first monotonic pressure verified via grid_contract.",
        }
        contracts["boundary_parameter_contract"] = {
            "status": "PASS" if checks["f_intern_parameter_rel"] <= SMOKE_REL_TOL else "FAIL",
            "checks": {
                "f_intern_boa_W_m2": checks["f_intern_boa_W_m2"],
                "expected_f_intern_W_m2": F_INT,
                "f_intern_parameter_rel": checks["f_intern_parameter_rel"],
            },
            "note": "F_intern is HELIOS internal-energy parameter contract, not instantaneous F_net(0).",
        }
        rocky_ok = (
            checks["rocky_surface_up_rel"] <= SMOKE_REL_TOL
            and checks["rocky_surface_net_rel"] <= SMOKE_REL_TOL
        )
        contracts["rocky_surface_radiative_bc"] = {
            "status": "PASS" if rocky_ok else "FAIL",
            "checks": {
                "f_up_boa_W_m2": checks["f_up_boa_W_m2"],
                "sigma_Tboa4_W_m2": sigma_tboa4,
                "f_down_boa_W_m2": checks["f_down_boa_W_m2"],
                "f_net_boa_W_m2": checks["f_net_boa_W_m2"],
                "rocky_surface_up_rel": checks["rocky_surface_up_rel"],
                "rocky_surface_net_rel": checks["rocky_surface_net_rel"],
            },
        }
        contracts["flux_decomposition_contract"] = {
            "status": "PASS" if decomp_rel <= SMOKE_DECOMP_REL_TOL else "FAIL",
            "checks": {"flux_decomposition_max_rel": decomp_rel},
            "note": "HELIOS integrated flux may differ slightly from F_up-F_down at roundoff.",
        }
        toa_ok = abs(checks["f_down_toa_W_m2"]) <= SMOKE_ABS_TOL_W_M2
        contracts["toa_thermal_contract"] = {
            "status": "PASS" if toa_ok else "FAIL",
            "checks": {"f_down_toa_W_m2": checks["f_down_toa_W_m2"]},
        }

        stage3 = _stage3_rocky_reference(
            grid,
            temperature_boa_k=t_boa,
            temperature_lay_k=np.asarray(prep["temperature_lay_k"], dtype=np.float64),
        )
        s3_up_rel = _rel_diff(float(up[0]), float(stage3["flux_up_W_m2"][0]), floor=sigma_tboa4)
        s3_net_rel = _rel_diff(float(net[0]), float(stage3["flux_net_W_m2"][0]), floor=sigma_tboa4)
        checks["stage3_vs_helios_up_rel"] = s3_up_rel
        checks["stage3_vs_helios_net_rel"] = s3_net_rel
        tol = SMOKE_REL_TOL
        if tolerances and "gates" in tolerances:
            tol = max(float(tolerances["gates"].get("flux_net_rel", tol)), tol)
        stage3_ok = s3_up_rel <= tol and s3_net_rel <= tol
        contracts["stage3_rocky_surface_parity"] = {
            "status": "PASS" if stage3_ok else "FAIL",
            "checks": {
                "stage3_vs_helios_up_rel": s3_up_rel,
                "stage3_vs_helios_net_rel": s3_net_rel,
                "stage3_lower_bc": "LowerTemperature(T_boa)",
            },
            "note": "Stage-3 uses HELIOS-equivalent rocky black-surface BC on the same asymmetric profile.",
        }

        required = [
            "helios_executable_smoke",
            "grid_contract",
            "tp_roundtrip_contract",
            "orientation_contract",
            "boundary_parameter_contract",
            "rocky_surface_radiative_bc",
            "flux_decomposition_contract",
            "toa_thermal_contract",
            "stage3_rocky_surface_parity",
        ]
        status = "PASS" if all(contracts[name]["status"] == "PASS" for name in required) else "FAIL"

    payload = {
        "case": "helios_contract_smoke_n8",
        "helios_commit": PINNED_HELIOS_COMMIT,
        "comparison_type": "contract_smoke_not_parity",
        "radiation_parity_scored": False,
        "status": status,
        "contracts": contracts,
        "checks": checks,
        "helios_runtime_config": runtime,
        "tp_checksum_sha256": _sha256(tp_path),
        "param_checksum_sha256": _sha256(param_path),
        "helios_flux_file": str(flux_path) if flux_path else None,
        **helios_track_status(
            adapter_contract=status,
            n96="NOT_RUN",
            n192="NOT_RUN",
        ),
        "note": (
            "Infrastructure smoke for HELIOS post-processing. F_intern=300 W/m^-2 is a "
            "parameter-contract check only. F_net(0) emerges from the frozen asymmetric profile."
        ),
        "invalid_legacy_assertions": [
            "f_net_boa_equals_f_intern",
        ],
    }
    if tolerances:
        payload["tolerances_source"] = str(tolerances)
    return payload


def rescore_existing(flux_path: Path, prep: dict, *, output: Path, tolerances: dict | None = None) -> dict:
    result = score_smoke(
        case_dir=flux_path.parent.parent
        if flux_path.parent.name == "helios_contract_smoke_n8"
        else flux_path.parent,
        flux_path=flux_path,
        prep=prep,
        tolerances=tolerances,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rescore-only", action="store_true")
    parser.add_argument("--helios-flux", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RESULTS / "helios_contract_smoke_n8.json")
    args = parser.parse_args()
    default_case = Path("/project/ls-heng/Bethany.Burt/helios_stage4_frozen/helios_contract_smoke_n8")
    case_dir = args.case_dir or default_case
    if not case_dir.exists():
        case_dir = RESULTS / "helios_contract_smoke_n8"

    tol_path = FIXTURES / "radiation_only_tolerances.json"
    tols = json.loads(tol_path.read_text()) if tol_path.exists() else None

    flux = args.helios_flux
    if flux is None:
        for candidate in (
            case_dir / "helios_contract_smoke_n8" / "helios_contract_smoke_n8_integrated_flux.dat",
            case_dir / "helios_contract_smoke_n8_integrated_flux.dat",
        ):
            if candidate.exists():
                flux = candidate
                break

    if args.rescore_only:
        prep_path = args.output.with_name("helios_contract_smoke_n8_prepare.json")
        if not prep_path.exists():
            raise FileNotFoundError(f"missing prepare payload: {prep_path}")
        prep = json.loads(prep_path.read_text())["prepare"]
        if flux is None:
            raise FileNotFoundError("rescore-only requires --helios-flux or existing integrated flux file")
        result = rescore_existing(flux, prep, output=args.output, tolerances=tols)
        print(json.dumps({"out": str(args.output), "status": result["status"]}, indent=2))
        return result

    prep = prepare_smoke_case(case_dir)
    if args.prepare_only:
        out = {"prepare": prep, "status": "PREPARED"}
        args.output.with_name("helios_contract_smoke_n8_prepare.json").write_text(json.dumps(out, indent=2) + "\n")
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return out

    result = score_smoke(case_dir=case_dir, flux_path=flux, prep=prep, tolerances=tols)
    result["prepare"] = prep
    args.output.with_name("helios_contract_smoke_n8_prepare.json").write_text(
        json.dumps({"prepare": prep, "status": "PREPARED"}, indent=2) + "\n"
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.output), "status": result["status"]}, indent=2))
    return result


if __name__ == "__main__":
    main()

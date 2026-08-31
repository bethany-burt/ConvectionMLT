"""Targeted Newton/Jv diagnostic bundle for the 1e-5 line-search factors.

Does not overwrite n384_implicit_rce.json. Does not launch coupled HELIOS.
Does not run five-check polish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    DEFAULT_DIFFUSIVITY,
    LowerNetInternalFlux,
    SteadyRCEConfig,
    TopIrradiation,
    active_interface_mask,
    evaluate_trial,
    jv_epsilon_ladder,
    mask_superadiabatic_excess,
    nested_analytic_opacity_spec,
)
from convection_mlt.radiation import SolveRoute

from rce_record import PHYSICAL_GATE, dumps, finalize_record, production_solver_config, verify_record_checksums
from solve_steady_rce import (
    LIVE_CHECKSUM,
    N192,
    N384,
    RESULTS,
    _complete_payload,
    _config_dict,
    _load_n96,
    _progress,
    _solve_record,
)

SRC_STEADY = ROOT.parent / "src" / "convection_mlt" / "steady_rce.py"
SRC_DRIVER = Path(__file__).resolve().parent / "solve_steady_rce.py"
BUNDLE = RESULTS / "steady_newton_diagnostic_bundle"
N96_OUT = RESULTS / "n96_steady_rce.json"
N192_OUT = RESULTS / "n192_steady_rce.json"
N384_OUT = RESULTS / "n384_steady_rce.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_cz_mask(n_layers: int, support: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_layers, dtype=np.float64)
    if bool(support[0]):
        mask[0] = 1.0
    for k in range(1, n_layers):
        if bool(support[k]):
            mask[k - 1] = 1.0
            mask[k] = 1.0
    return mask


def _diagnostic_config(**overrides) -> SteadyRCEConfig:
    payload = dict(
        flux_flatness_tolerance=PHYSICAL_GATE,
        tendency_tolerance=PHYSICAL_GATE,
        max_newton=8,
        max_mask_outer=1,
        gmres_maxiter=48,
        gmres_restart=32,
        fd_rel=1.0e-6,
        subspace_fd_rel=1.0e-8,
        max_step_rel=0.05,
        use_subspace=True,
        use_gmres=True,
        reject_branch_crossing=True,
        min_line_search_factor=1.0e-8,
        max_line_search=20,
        armijo_c=1.0e-4,
        max_tendency_increase=2.0,
    )
    payload.update(overrides)
    return SteadyRCEConfig(**payload)


def _write_complete(path: Path, res, spec, solver, source: dict, summary: dict, cfg: SteadyRCEConfig) -> dict:
    payload = _complete_payload(res, spec, solver, source, summary, cfg)
    path.write_text(dumps(finalize_record(payload)))
    return payload


def _solver_settings(cfg: SteadyRCEConfig) -> dict:
    return {
        "purpose": (
            "Reduced live-MLT subspace Newton with Armijo merit search. "
            "Not a coupled-HELIOS claim and not a gated N=384 spatial claim."
        ),
        "physical_gate": PHYSICAL_GATE,
        "live_n384_checksum": LIVE_CHECKSUM,
        "do_not_overwrite": "n384_implicit_rce.json",
        "coupled_helios_rce_status": "NOT_RUN",
        "five_check_polish": "NOT_RUN",
        "steady_config": _config_dict(cfg),
        "finite_difference_jv": {
            "subspace_formula": (
                "centred live MLT: Jv ≈ [r(h+eps d) − r(h−eps d)] / (2 eps), "
                "eps = subspace_fd_rel * ||h|| / ||d||, default subspace_fd_rel=1e-8"
            ),
            "gmres_fallback_formula": (
                "forward lagged F_conv: Jv ≈ [r(h+eps d) − r0] / eps, "
                "eps = fd_rel * ||h|| / ||d||, default fd_rel=1e-6"
            ),
            "reject_branch_crossing": cfg.reject_branch_crossing,
            "h_floor": cfg.h_floor,
            "line_search": "Armijo Φ=½||r||₂² with Φ(h+αΔh)≤(1−cα)Φ(h); tendency ≤ max(current, gate)×max_tendency_increase",
            "variable_scaling": "GMRES unknown v is dimensionless; physical step is dh = h_scale * v with h_scale = max(|h|, h_floor)",
        },
        "routines": {
            "live_F_conv": "convection_mlt.steady_rce.live_convective_flux / evaluate_trial (frozen_flux_conv=None)",
            "active_mask": "convection_mlt.steady_rce.active_interface_mask / interface_support_from_regions",
            "total_flux_residual": "convection_mlt.steady_rce.flux_flatness_residual / flux_metrics / evaluate_trial",
            "superadiabatic_excess": "convection_mlt.steady_rce.mask_superadiabatic_excess",
            "jv_epsilon_ladder": "convection_mlt.steady_rce.jv_epsilon_ladder",
        },
        "iteration_history_fields": [
            "flux_flatness_before",
            "flux_flatness_after",
            "residual_before",
            "residual_after",
            "gmres_iters",
            "gmres_residual_norm",
            "gmres_rhs_norm",
            "gmres_rtol",
            "gmres_success",
            "line_search_factor",
            "line_search_reason",
            "step_rel_h_newton",
            "step_rel_h_accepted",
            "step_rel_T_accepted",
            "mask_before",
            "mask_after",
            "min_superadiabatic_excess_active",
            "max_superadiabatic_excess_inactive",
            "rcb_active_distance_to_threshold",
            "rcb_inactive_distance_to_threshold",
            "n_branch_crossings",
            "jv_n_branch_crossings",
            "merit_before",
            "merit_after",
            "fd_rel",
            "jv_eps_used",
            "h_scale_rms",
        ],
        "ladder_interpretation": (
            "jv_stable uses only rungs with from_fd_rel ≤ 1e-6. Coherent CZ and "
            "residual-scaled live Jv approach a local derivative in that "
            "neighbourhood; random high-frequency directions need not. "
            "A large full-ladder max pairwise change at fd_rel=1e-3 is "
            "nonlinearity, not proof that no coupled Jacobian exists."
        ),
    }


def _residual_fn(
    spec,
    enthalpy_ref: np.ndarray,
    support: np.ndarray,
    *,
    lagged_flux: np.ndarray | None,
):
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    f_int = spec.f_int
    f_scale = abs(f_int)

    def trial_at(enthalpy: np.ndarray):
        return evaluate_trial(
            grid,
            enthalpy,
            spec.physics(),
            thermo,
            spec.opacity(),
            grid.pressure_centres,
            TopIrradiation(spec.f_irr),
            LowerNetInternalFlux(spec.f_int),
            ConstantGravity(spec.gravity),
            f_int=f_int,
            f_scale=f_scale,
            frozen_support=support,
            diffusivity_factor=DEFAULT_DIFFUSIVITY,
            radiation_route=SolveRoute.THOMAS,
            frozen_flux_conv=lagged_flux,
        )

    def residual_at(enthalpy: np.ndarray):
        trial = trial_at(enthalpy)
        return None if trial is None else trial.residual

    r0 = residual_at(enthalpy_ref)
    if r0 is None:
        raise RuntimeError("reference residual evaluation left the domain")
    return residual_at, trial_at, r0


def _run_ladder(n384: dict) -> dict:
    spec = nested_analytic_opacity_spec(384)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    h = np.asarray(n384["enthalpy"], dtype=np.float64)
    trial = evaluate_trial(
        grid,
        h,
        spec.physics(),
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        ConstantGravity(spec.gravity),
        f_int=spec.f_int,
        f_scale=abs(spec.f_int),
        frozen_support=None,
        diffusivity_factor=DEFAULT_DIFFUSIVITY,
        radiation_route=SolveRoute.THOMAS,
    )
    if trial is None:
        raise RuntimeError("live N=384 residual evaluation left the domain")
    support = active_interface_mask(grid.n_layers, trial.closure, solver)
    h_scale = np.maximum(np.abs(h), 1.0e-30)
    layer_cz = _layer_cz_mask(grid.n_layers, support)
    layer_rad = 1.0 - layer_cz
    rng = np.random.default_rng(384)
    directions = {
        "cz_relative": h_scale * layer_cz,
        "radiative_zone": h_scale * layer_rad,
        "residual_scaled": h_scale * np.asarray(trial.residual, dtype=np.float64),
        "random_seeded": h_scale * rng.standard_normal(h.size),
    }
    lagged = np.asarray(trial.flux_conv, dtype=np.float64)
    report: dict[str, object] = {
        "n_layers": 384,
        "source_profile_checksum_sha256": n384.get("profile_checksum_sha256"),
        "source_flux_flatness": trial.flux_flatness,
        "source_tendency_norm": trial.tendency_norm,
        "convective_regions": n384.get("convective_regions"),
        "n_active_interfaces": int(np.count_nonzero(support)),
        "superadiabatic_excess": mask_superadiabatic_excess(trial.closure, support, solver),
        "h_norm": float(np.linalg.norm(h)),
        "h_scale_rms": float(np.sqrt(np.mean(np.square(h_scale)))),
        "directions": {},
    }
    for name, direction in directions.items():
        dnorm = float(np.linalg.norm(direction))
        if dnorm == 0.0:
            report["directions"][name] = {"ok": False, "reason": "zero_direction"}
            continue
        direction = direction / dnorm
        print(f"  Jv ladder direction={name}", flush=True)
        live_fn, live_trial, live_r0 = _residual_fn(spec, h, support, lagged_flux=None)
        lagged_fn, lagged_trial, lagged_r0 = _residual_fn(spec, h, support, lagged_flux=lagged)
        live = jv_epsilon_ladder(
            live_fn,
            h,
            live_r0,
            direction,
            trial_at=live_trial,
            frozen_support=support,
            solver=solver,
            n_layers=grid.n_layers,
        )
        lagged_out = jv_epsilon_ladder(
            lagged_fn,
            h,
            lagged_r0,
            direction,
            trial_at=lagged_trial,
            frozen_support=support,
            solver=solver,
            n_layers=grid.n_layers,
        )
        report["directions"][name] = {
            "direction_norm_before_unit": dnorm,
            "live_mlt": live,
            "lagged_f_conv": lagged_out,
            "live_jv_stable": live["jv_stable"],
            "lagged_jv_stable": lagged_out["jv_stable"],
            "live_jv_stable_full_ladder": live["jv_stable_full_ladder"],
            "lagged_jv_stable_full_ladder": lagged_out["jv_stable_full_ladder"],
            "live_max_pairwise_rel_two_change": live["max_pairwise_rel_two_change"],
            "lagged_max_pairwise_rel_two_change": lagged_out["max_pairwise_rel_two_change"],
            "live_max_pairwise_rel_two_change_fine": live["max_pairwise_rel_two_change_fine"],
            "lagged_max_pairwise_rel_two_change_fine": lagged_out["max_pairwise_rel_two_change_fine"],
        }
        print(
            f"    live fine_stable={live['jv_stable']} "
            f"fine_rel={live['max_pairwise_rel_two_change_fine']:.3g} "
            f"full_rel={live['max_pairwise_rel_two_change']:.3g}; "
            f"lagged fine_stable={lagged_out['jv_stable']} "
            f"fine_rel={lagged_out['max_pairwise_rel_two_change_fine']:.3g}",
            flush=True,
        )
    live_max = [
        float(d["live_max_pairwise_rel_two_change"])
        for d in report["directions"].values()
        if isinstance(d, dict) and d.get("live_max_pairwise_rel_two_change") is not None
    ]
    lagged_max = [
        float(d["lagged_max_pairwise_rel_two_change"])
        for d in report["directions"].values()
        if isinstance(d, dict) and d.get("lagged_max_pairwise_rel_two_change") is not None
    ]
    live_fine = [
        float(d["live_max_pairwise_rel_two_change_fine"])
        for d in report["directions"].values()
        if isinstance(d, dict) and d.get("live_max_pairwise_rel_two_change_fine") is not None
    ]
    lagged_fine = [
        float(d["lagged_max_pairwise_rel_two_change_fine"])
        for d in report["directions"].values()
        if isinstance(d, dict) and d.get("lagged_max_pairwise_rel_two_change_fine") is not None
    ]
    report["summary"] = {
        "any_live_unstable_fine": any(
            not bool(d.get("live_jv_stable"))
            for d in report["directions"].values()
            if isinstance(d, dict)
        ),
        "any_lagged_unstable_fine": any(
            not bool(d.get("lagged_jv_stable"))
            for d in report["directions"].values()
            if isinstance(d, dict)
        ),
        "max_live_pairwise_rel_full": max(live_max) if live_max else float("nan"),
        "max_lagged_pairwise_rel_full": max(lagged_max) if lagged_max else float("nan"),
        "max_live_pairwise_rel_fine": max(live_fine) if live_fine else float("nan"),
        "max_lagged_pairwise_rel_fine": max(lagged_fine) if lagged_fine else float("nan"),
    }
    return report


def _index_bundle(bundle: Path) -> dict:
    files = {}
    for path in sorted(p for p in bundle.iterdir() if p.is_file() and p.name != "bundle_index.json"):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return {
        "purpose": (
            "Steady Newton–Krylov diagnostic bundle for the N=384 1e-5 "
            "line-search factors. Not a coupled-HELIOS claim and not a "
            "gated N=384 spatial claim."
        ),
        "live_n384_checksum": LIVE_CHECKSUM,
        "live_n384_file": "n384_implicit_rce.json",
        "stale_checkpoint_excluded": "n384_implicit_rce.checkpoint.json",
        "n192_implicit_record_excluded": True,
        "coupled_helios_rce_status": "NOT_RUN",
        "five_check_polish": "NOT_RUN",
        "files": files,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--skip-n384-newton", action="store_true")
    args = parser.parse_args(argv)

    n384 = json.loads(N384.read_text())
    verify_record_checksums(n384)
    stored = n384.get("profile_checksum_sha256")
    if stored != LIVE_CHECKSUM:
        raise AssertionError(
            f"N=384 live checksum {stored} != expected gated checksum {LIVE_CHECKSUM}"
        )
    print(
        f"N=384 live record checksum ok: {stored} "
        f"steps={n384.get('steps_accepted')} flatness={n384.get('flux_flatness')}",
        flush=True,
    )

    BUNDLE.mkdir(parents=True, exist_ok=True)
    cfg = _diagnostic_config()
    settings = _solver_settings(cfg)
    (RESULTS / "steady_newton_solver_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (BUNDLE / "solver_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    shutil.copy2(SRC_STEADY, BUNDLE / "steady_rce.py")
    shutil.copy2(SRC_DRIVER, BUNDLE / "solve_steady_rce.py")
    shutil.copy2(Path(__file__).resolve(), BUNDLE / "steady_newton_diagnostics.py")
    shutil.copy2(N384, BUNDLE / "n384_implicit_rce.json")

    n96 = _load_n96()
    n192 = json.loads(N192.read_text())
    verify_record_checksums(n192)

    report: dict = {
        "live_n384_checksum": stored,
        "coupled_helios_rce_status": "NOT_RUN",
        "five_check_polish": "NOT_RUN",
        "bundle": str(BUNDLE),
    }

    print("\n=== N=96 complete steady result (physical gate) ===", flush=True)
    res96, spec96, solver96, sum96 = _solve_record(96, n96, cfg)
    payload96 = _write_complete(N96_OUT, res96, spec96, solver96, n96, sum96, cfg)
    shutil.copy2(N96_OUT, BUNDLE / "n96_steady_rce.json")
    report["n96"] = {
        "status": sum96["status"],
        "newton_iterations": sum96["newton_iterations"],
        "max_rel_T": sum96["max_rel_T"],
        "flux_flatness": sum96["flux_flatness"],
        "history_len": len(sum96["history"]),
        "profile_checksum_sha256": payload96["profile_checksum_sha256"],
    }

    print("\n=== N=192 complete steady result (physical gate) ===", flush=True)
    res192, spec192, solver192, sum192 = _solve_record(192, n192, cfg)
    payload192 = _write_complete(N192_OUT, res192, spec192, solver192, n192, sum192, cfg)
    shutil.copy2(N192_OUT, BUNDLE / "n192_steady_rce.json")
    report["n192"] = {
        "status": sum192["status"],
        "newton_iterations": sum192["newton_iterations"],
        "max_rel_T": sum192["max_rel_T"],
        "flux_flatness": sum192["flux_flatness"],
        "history_len": len(sum192["history"]),
        "profile_checksum_sha256": payload192["profile_checksum_sha256"],
    }

    if not args.skip_ladder:
        print("\n=== N=384 Jv epsilon ladder ===", flush=True)
        ladder = _run_ladder(n384)
        (RESULTS / "jv_epsilon_ladder.json").write_text(json.dumps(ladder, indent=2, allow_nan=True) + "\n")
        (BUNDLE / "jv_epsilon_ladder.json").write_text(json.dumps(ladder, indent=2, allow_nan=True) + "\n")
        report["jv_epsilon_ladder"] = ladder["summary"]

    if not args.skip_n384_newton:
        print("\n=== N=384 diagnostic Newton (live-MLT subspace, Armijo merit) ===", flush=True)
        res384, spec384, solver384, sum384 = _solve_record(384, n384, cfg)
        payload384 = _write_complete(N384_OUT, res384, spec384, solver384, n384, sum384, cfg)
        shutil.copy2(N384_OUT, BUNDLE / "n384_steady_rce.json")
        alphas = [rec["line_search_factor"] for rec in sum384["history"]]
        reasons = [rec["line_search_reason"] for rec in sum384["history"]]
        report["n384"] = {
            "status": sum384["status"],
            "newton_iterations": sum384["newton_iterations"],
            "max_rel_T": sum384["max_rel_T"],
            "flux_flatness": sum384["flux_flatness"],
            "tendency_norm": sum384["tendency_norm"],
            "line_search_factors": alphas,
            "line_search_reasons": reasons,
            "history_len": len(sum384["history"]),
            "profile_checksum_sha256": payload384["profile_checksum_sha256"],
            "source_profile_checksum_sha256": stored,
            "live_record_unchanged": True,
        }
        print(
            json.dumps(
                {
                    "status": sum384["status"],
                    "flatness": sum384["flux_flatness"],
                    "newton": sum384["newton_iterations"],
                    "alphas": alphas,
                    "reasons": reasons,
                    "dT": sum384["max_rel_T"],
                },
                indent=2,
            ),
            flush=True,
        )

    index = _index_bundle(BUNDLE)
    (BUNDLE / "bundle_index.json").write_text(json.dumps(index, indent=2) + "\n")
    report["bundle_files"] = list(index["files"])
    (RESULTS / "steady_newton_diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True)[:4000], flush=True)
    return report


if __name__ == "__main__":
    main()

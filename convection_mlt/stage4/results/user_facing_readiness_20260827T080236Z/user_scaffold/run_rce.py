#!/usr/bin/env python3
"""Minimal user-facing nested-τ RCE runner (simple-solver demonstration).

Usage:
  PYTHONPATH=src python -m convection_mlt.user_run --config path/to/config.json
  PYTHONPATH=src python stage4/user/run_rce.py --config stage4/user/example_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    RCERoute,
    TopIrradiation,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from rce_record import (
    PHYSICAL_GATE,
    dumps,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)


DEFAULT_CONFIG = {
    "n_layers": 96,
    "alpha": 1.0,
    "f_int": 300.0,
    "f_irr": 0.0,
    "max_steps": 20000,
    "dt_accuracy": 2500.0,
    "gate": PHYSICAL_GATE,
    "seed": "radiative_convective",
    "output": "rce_result.json",
    "notes": (
        "Piecewise-isothermal cell-centred radiation on nested-τ grids; "
        "finite MLT. Not equivalent at finite resolution to HELIOS "
        "non-isothermal within-layer source + convective adjustment."
    ),
}


def load_config(path: Path | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        user = json.loads(path.read_text())
        cfg.update(user)
    return cfg


def run(cfg: dict) -> dict:
    spec = nested_analytic_opacity_spec(
        int(cfg["n_layers"]),
        alpha=float(cfg["alpha"]),
        f_int=float(cfg.get("f_int", 300.0)),
        f_irr=float(cfg.get("f_irr", 0.0)),
    )
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    t0 = radiative_convective_initial_temperature(
        grid, spec.opacity(), thermo, spec.f_int, spec.f_irr
    )
    rce_cfg = production_rce_config(
        max_steps=int(cfg["max_steps"]),
        dt_accuracy=float(cfg["dt_accuracy"]),
        gate=float(cfg["gate"]),
    )
    res = solve_adaptive_rce(
        grid,
        t0,
        spec.physics(),
        production_solver_config(),
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=rce_cfg,
    )
    return serialize_rce_result(
        res,
        spec,
        pressure_centres=grid.pressure_centres,
        pressure_edges=grid.pressure_edges,
        solver=production_solver_config(),
        rce_config=rce_cfg,
        extra={"user_config": cfg, "notes": cfg.get("notes")},
    )


def write_plots(record: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    p = np.asarray(record["pressure_centres"], dtype=float)
    t = np.asarray(record["temperature"], dtype=float)
    fr = np.asarray(record["flux_rad"], dtype=float)
    fc = np.asarray(record["flux_conv"], dtype=float)
    ft = np.asarray(record["flux_total"], dtype=float)
    written = []

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    ax.semilogy(t, p)
    ax.invert_yaxis()
    ax.set_xlabel("T [K]")
    ax.set_ylabel("P [Pa]")
    ax.set_title(
        f"N={record.get('n_layers')}  α={((record.get('physics_config') or {}).get('alpha'))}  "
        f"RCB={record.get('primary_rcb_log10p')}"
    )
    path = out_dir / "tp_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    pe = np.asarray(record.get("pressure_edges") or p, dtype=float)
    if pe.size == fr.size:
        ax.semilogy(fr, pe, label="F_rad")
        ax.semilogy(fc, pe, label="F_conv")
        ax.semilogy(ft, pe, label="F_total")
        ax.invert_yaxis()
        ax.set_xlabel("Flux [W m$^{-2}$]")
        ax.set_ylabel("P [Pa]")
        ax.legend()
        path = out_dir / "flux_profile.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--write-example-config", type=Path, default=None)
    ap.add_argument("--plot-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.write_example_config is not None:
        args.write_example_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_example_config.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
        print(f"wrote {args.write_example_config}")
        return
    cfg = load_config(args.config)
    rec = run(cfg)
    out = Path(cfg["output"])
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(rec))
    print(
        json.dumps(
            {
                "output": str(out),
                "status": rec.get("status"),
                "flux_flatness": rec.get("flux_flatness"),
                "primary_rcb_log10p": rec.get("primary_rcb_log10p"),
                "physically_gated": (
                    rec.get("status") == "converged"
                    and float(rec.get("flux_flatness") or 1) <= float(cfg["gate"])
                    and float(rec.get("tendency_norm") or 1) <= float(cfg["gate"])
                ),
            },
            indent=2,
        )
    )
    plot_dir = args.plot_dir
    if plot_dir is None:
        plot_dir = out.with_suffix("").parent / (out.stem + "_plots")
    written = write_plots(rec, plot_dir)
    print(json.dumps({"plots": written}, indent=2))


if __name__ == "__main__":
    main()

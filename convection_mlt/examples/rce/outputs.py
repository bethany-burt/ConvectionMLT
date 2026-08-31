"""Reproducible run-directory writers for the user-facing RCE runner."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np

from convection_mlt.metadata import git_commit, git_dirty
from convection_mlt.production_rce import (
    PHYSICAL_GATE,
    ProductionRCERun,
    evaluate_physical_gates,
    production_thermo,
)

# serialize via stage4 experiments helper
import sys

_PKG = Path(__file__).resolve().parents[2]
_EXP = _PKG / "stage4" / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
from rce_record import dumps, serialize_rce_result  # noqa: E402

from load_cfg import ValidatedConfig, _scalar_thermo


class OutputDirError(RuntimeError):
    pass


def prepare_output_dir(path: Path, *, force: bool) -> Path:
    path = path.resolve()
    if path.exists():
        if any(path.iterdir()):
            if not force:
                raise OutputDirError(
                    f"output directory {path} is non-empty; pass --force to overwrite"
                )
            for child in path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: TextIO = path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        self._fh.write(line)
        self._fh.flush()
        print(msg, flush=True)

    def close(self) -> None:
        self._fh.close()


def _pa_to_bar(p_pa: np.ndarray) -> np.ndarray:
    return np.asarray(p_pa, dtype=np.float64) / 1.0e5


def write_profiles(
    out_dir: Path,
    run: ProductionRCERun,
) -> dict[str, str]:
    p_c = run.pressure_centres
    p_e = run.pressure_edges
    t0 = run.temperature_initial
    t = np.asarray(run.result.final_state.temperature, dtype=np.float64)
    fr = np.asarray(run.result.final_flux_rad, dtype=np.float64)
    fc = np.asarray(run.result.final_flux_conv, dtype=np.float64)
    ft = np.asarray(run.result.final_flux_total, dtype=np.float64)
    nabla = run.nabla
    nabla_ad = run.nabla_ad
    delta = run.delta_nabla

    centres = out_dir / "profiles_centres.csv"
    with centres.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["P_Pa", "P_bar", "T_init_K", "T_final_K"])
        for i in range(p_c.size):
            w.writerow([float(p_c[i]), float(p_c[i] / 1e5), float(t0[i]), float(t[i])])

    interfaces = out_dir / "profiles_interfaces.csv"
    n_edge = p_e.size
    if fr.size != n_edge or nabla.size != n_edge or nabla_ad.size != n_edge:
        raise RuntimeError(
            f"interface length mismatch: P_edges={n_edge}, "
            f"flux={fr.size}, nabla={nabla.size}, nabla_ad={nabla_ad.size}"
        )
    with interfaces.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "P_Pa",
                "P_bar",
                "F_rad_W_m2",
                "F_conv_W_m2",
                "F_total_W_m2",
                "nabla",
                "nabla_ad",
                "delta_nabla",
            ]
        )
        for i in range(n_edge):
            w.writerow(
                [
                    float(p_e[i]),
                    float(p_e[i] / 1e5),
                    float(fr[i]),
                    float(fc[i]),
                    float(ft[i]),
                    float(nabla[i]),
                    float(nabla_ad[i]),
                    float(delta[i]),
                ]
            )

    npz = out_dir / "profiles.npz"
    np.savez(
        npz,
        pressure_centres_Pa=p_c,
        pressure_edges_Pa=p_e,
        T_init_K=t0,
        T_final_K=t,
        F_rad_W_m2=fr,
        F_conv_W_m2=fc,
        F_total_W_m2=ft,
        nabla=nabla,
        nabla_ad=nabla_ad,
        delta_nabla=delta,
    )
    return {
        "profiles_centres.csv": str(centres),
        "profiles_interfaces.csv": str(interfaces),
        "profiles.npz": str(npz),
    }


def write_convergence_csv(out_dir: Path, run: ProductionRCERun) -> Path:
    path = out_dir / "convergence.csv"
    time_col = "physical_time_s" if run.prescribed_dt is not None else "pseudo_time_s"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["step", "phase", time_col, "dt_s", "flux_flatness", "tendency_norm"]
        )
        for row in run.convergence_log:
            w.writerow(
                [
                    row.step,
                    row.phase,
                    row.time_s,
                    row.dt,
                    row.flux_flatness,
                    row.tendency_norm,
                ]
            )
    return path


def _plot_temperature(ax, run: ProductionRCERun, *, p_c_bar: np.ndarray) -> None:
    t0 = run.temperature_initial
    t = np.asarray(run.result.final_state.temperature, dtype=np.float64)
    rcb = run.result.primary_rcb_log10p
    ax.semilogy(t0, p_c_bar, label="initial", color="0.55")
    ax.semilogy(t, p_c_bar, label="final", color="C0")
    if rcb is not None:
        ax.axhline(
            10 ** (float(rcb) - 5.0),
            color="C3",
            ls="--",
            label=f"RCB log10P={rcb:.3f}",
        )
    ax.invert_yaxis()
    ax.set_xlabel("T [K]")
    ax.set_ylabel("P [bar]")
    ax.set_title("Temperature")
    ax.legend(fontsize=8)


def _plot_fluxes(ax, run: ProductionRCERun, *, p_e_bar: np.ndarray) -> None:
    fr = np.asarray(run.result.final_flux_rad, dtype=np.float64)
    fc = np.asarray(run.result.final_flux_conv, dtype=np.float64)
    ft = np.asarray(run.result.final_flux_total, dtype=np.float64)
    f_int = float(run.spec.f_int)
    f_irr = float(run.spec.f_irr)
    ax.semilogy(fr, p_e_bar, label=r"$F_{\rm rad}$")
    ax.semilogy(fc, p_e_bar, label=r"$F_{\rm conv}$")
    ax.semilogy(ft, p_e_bar, label=r"$F_{\rm total}$")
    ax.axvline(f_int, color="k", ls=":", label=r"$F_{\rm int}$")
    if abs(f_irr) > 0.0:
        ax.axvline(f_irr, color="0.4", ls="--", label=r"$F_{\rm irr}$")
    ax.invert_yaxis()
    ax.set_xlabel(r"Flux [W m$^{-2}$]")
    ax.set_ylabel("P [bar]")
    ax.set_title("Fluxes (positive = upward)")
    ax.legend(fontsize=8)


def _plot_gradients(ax, run: ProductionRCERun, *, p_e_bar: np.ndarray) -> None:
    ax.semilogy(run.nabla, p_e_bar, label=r"$\nabla$")
    ax.semilogy(run.nabla_ad, p_e_bar, label=r"$\nabla_{\rm ad}$")
    ax.semilogy(np.maximum(run.delta_nabla, 0.0), p_e_bar, label=r"$\Delta\nabla$")
    ax.invert_yaxis()
    ax.set_xlabel("Gradient")
    ax.set_ylabel("P [bar]")
    ax.set_title("Thermal gradients")
    ax.legend(fontsize=8)


def _plot_convergence(ax, run: ProductionRCERun, *, gate: float) -> None:
    if run.convergence_log:
        steps = [r.step for r in run.convergence_log]
        flat = [max(r.flux_flatness, 1e-16) for r in run.convergence_log]
        tend = [max(r.tendency_norm, 1e-16) for r in run.convergence_log]
        ax.semilogy(steps, flat, label="flux flatness")
        ax.semilogy(steps, tend, label="tendency")
    ax.axhline(gate, color="C3", ls="--", label=f"gate={gate:g}")
    time_note = "physical Δt" if run.prescribed_dt is not None else "pseudo-time"
    ax.set_xlabel(f"accepted step ({time_note})")
    ax.set_ylabel("residual")
    ax.set_title("Convergence history")
    ax.legend(fontsize=8)


def write_summary_figure(
    out_dir: Path,
    run: ProductionRCERun,
    cfg: ValidatedConfig,
    *,
    gate: float = PHYSICAL_GATE,
) -> Path | None:
    panels: list[tuple[str, Callable[..., None]]] = []
    if cfg.plot_temperature:
        panels.append(("temperature", _plot_temperature))
    if cfg.plot_fluxes:
        panels.append(("fluxes", _plot_fluxes))
    if cfg.plot_gradients:
        panels.append(("gradients", _plot_gradients))
    if cfg.plot_convergence:
        panels.append(("convergence", _plot_convergence))

    if not panels:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_c_bar = _pa_to_bar(run.pressure_centres)
    p_e_bar = _pa_to_bar(run.pressure_edges)

    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.25 * ncols, 4.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (name, plot_fn) in zip(axes_flat, panels):
        if name == "temperature":
            plot_fn(ax, run, p_c_bar=p_c_bar)
        elif name in {"fluxes", "gradients"}:
            plot_fn(ax, run, p_e_bar=p_e_bar)
        else:
            plot_fn(ax, run, gate=gate)

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"N={run.spec.n_layers}  α={run.spec.alpha}  procedure={run.procedure}",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "figure_summary.png"
    fig.savefig(path, dpi=cfg.figure_dpi)
    plt.close(fig)
    return path


def build_result_record(run: ProductionRCERun, cfg: ValidatedConfig) -> dict[str, Any]:
    thermo = production_thermo(cfg.x_he)
    t_ref = 1500.0
    eos = "ConstantH2Thermo" if cfg.x_he == 0.0 else "h2_he_mixture"
    extra = {
        "temperature_initial": run.temperature_initial.tolist(),
        "nabla": run.nabla.tolist(),
        "nabla_ad": run.nabla_ad.tolist(),
        "delta_nabla": run.delta_nabla.tolist(),
        "phases": run.phases,
        "procedure": run.procedure,
        "prescribed_dt": run.prescribed_dt,
        "user_config_checksum_sha256": cfg.config_checksum_sha256,
        "actual_integrator": (
            "discrete_rz_t_rcb_finite_mlt_then_five_check_pseudotime"
            if run.procedure == "production"
            else "adaptive_only"
        ),
        "time_base": "physical" if run.prescribed_dt is not None else "pseudo-time",
        "eos": eos,
        "x_he": cfg.x_he,
        "nabla_ad_scalar": _scalar_thermo(thermo.nabla_ad_at(t_ref)),
    }
    return serialize_rce_result(
        run.result,
        run.spec,
        pressure_centres=run.pressure_centres,
        pressure_edges=run.pressure_edges,
        solver=run.solver,
        rce_config=run.rce_config_last,
        extra=extra,
    )


def write_status(
    out_dir: Path,
    *,
    cfg: ValidatedConfig,
    record: dict[str, Any],
    gates,
    verdict: str,
    started_utc: str,
    ended_utc: str,
    wall_s: float,
) -> Path:
    require_topo = abs(cfg.f_irr) <= 1.0e-15
    status = {
        "verdict": verdict,
        "convergence": gates.convergence_ok,
        "topology_ok": gates.topology_ok,
        "gate": cfg.gate,
        "require_bottom_connected_cz": require_topo,
        "gates": gates.as_dict,
        "flux_flatness": gates.flux_flatness,
        "tendency_norm": gates.tendency_norm,
        "primary_rcb_log10p": record.get("primary_rcb_log10p"),
        "convective_regions": record.get("convective_regions"),
        "detached_convective_regions": record.get("detached_convective_regions"),
        "validation_envelope": cfg.envelope_status,
        "validation_envelope_warnings": list(cfg.envelope_warnings),
        "config_checksum_sha256": cfg.config_checksum_sha256,
        "config_path": cfg.config_path,
        "profile_checksum_sha256": record.get("profile_checksum_sha256"),
        "code_git_commit": git_commit(),
        "code_git_dirty": git_dirty(),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "wall_time_s": wall_s,
        "procedure": cfg.procedure,
        "n_layers": cfg.n_layers,
        "alpha": cfg.alpha,
        "seed": cfg.seed,
        "x_he": cfg.x_he,
    }
    path = out_dir / "status.json"
    path.write_text(json.dumps(status, indent=2) + "\n")
    return path


def write_run_artifacts(
    out_dir: Path,
    cfg: ValidatedConfig,
    run: ProductionRCERun,
    *,
    started_utc: str,
    ended_utc: str,
    wall_s: float,
) -> tuple[str, dict[str, Any]]:
    snapshot = cfg.to_snapshot()
    (out_dir / "input_resolved.json").write_text(json.dumps(snapshot, indent=2) + "\n")

    record = build_result_record(run, cfg)
    require_topo = abs(cfg.f_irr) <= 1.0e-15
    gates = evaluate_physical_gates(
        record, gate=cfg.gate, require_bottom_connected_cz=require_topo
    )
    if gates.convergence_ok and (gates.topology_ok or not require_topo):
        verdict = "CONVERGED"
    else:
        verdict = "NOT CONVERGED"

    if cfg.write_result_json:
        (out_dir / "result.json").write_text(dumps(record))
    if cfg.write_profiles:
        write_profiles(out_dir, run)
    if cfg.write_convergence:
        write_convergence_csv(out_dir, run)
    if cfg.write_figure:
        write_summary_figure(out_dir, run, cfg, gate=cfg.gate)
    if cfg.write_status:
        write_status(
            out_dir,
            cfg=cfg,
            record=record,
            gates=gates,
            verdict=verdict,
            started_utc=started_utc,
            ended_utc=ended_utc,
            wall_s=wall_s,
        )
    return verdict, record

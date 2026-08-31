"""Load and validate VULCAN-style Python case files for the RCE runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from convection_mlt.production_rce import (
    DEFAULT_ALPHA,
    DEFAULT_F_INT,
    DEFAULT_F_IRR,
    DEFAULT_GRAVITY,
    DEFAULT_P_BOTTOM,
    DEFAULT_P_TOP,
    DEFAULT_X_HE,
    PHYSICAL_GATE,
    ProductionControls,
    production_thermo,
    validation_envelope,
)

ALLOWED_N_LAYERS = frozenset({96, 192, 384})  # nested master family (formal spatial gate)
MIN_N_LAYERS = 4
MAX_N_LAYERS = 4096
ALLOWED_SEEDS = frozenset({"radiative_convective", "radiative_equilibrium"})
ALLOWED_PROCEDURES = frozenset({"production", "adaptive_only"})
OPACITY_MODEL = "analytic_grey_powerlaw"

FORBIDDEN_USER_VARS = frozenset({"gate"})


def _scalar_thermo(value: Any) -> float:
    """Extract a float from thermo provider output (0-d or 1-element arrays)."""
    arr = np.asarray(value, dtype=np.float64)
    return float(arr.item() if arr.ndim == 0 else arr.ravel()[0])

FLUX_SIGN_CONVENTION = {
    "f_int": "Net upward internal flux imposed at the bottom boundary [W m^-2].",
    "f_irr": "Downward irradiation at the top boundary [W m^-2].",
    "profile_fluxes": (
        "Positive flux is upward net energy transport; "
        "F_total = F_rad + F_conv on interfaces."
    ),
}

BAR_TO_PA = 1.0e5
DEFAULT_P_BOTTOM_BAR = DEFAULT_P_BOTTOM / BAR_TO_PA  # 10 bar
DEFAULT_P_TOP_BAR = DEFAULT_P_TOP / BAR_TO_PA        # 1e-5 bar

DEFAULTS: dict[str, Any] = {
    "n_layers": 96,
    "p_bottom": DEFAULT_P_BOTTOM_BAR,
    "p_top": DEFAULT_P_TOP_BAR,
    "gravity": DEFAULT_GRAVITY,
    "x_he": DEFAULT_X_HE,
    "f_int": DEFAULT_F_INT,
    "f_irr": DEFAULT_F_IRR,
    "alpha": DEFAULT_ALPHA,
    "seed": "radiative_convective",
    "opacity_model": OPACITY_MODEL,
    "procedure": "production",
    "max_steps_live_polish": 200,
    "max_steps_continuation": 500,
    "max_recovery_cycles": 2,
    "dt_accuracy_s": 50000.0,
    "dt_hold_init_s": 18415.0,
    "continuation_dt_accuracy_s": 2500.0,
    "prescribed_dt_s": None,
    "max_steps_adaptive_only": 20000,
    "output_dir": "examples/rce/runs",
    "out_name": "",
    "overwrite": False,
    "write_profiles": True,
    "write_convergence": True,
    "write_result_json": True,
    "write_status": True,
    "write_figure": True,
    "figure_dpi": 150,
    "plot_temperature": True,
    "plot_fluxes": True,
    "plot_gradients": True,
    "plot_convergence": True,
}

USER_VAR_NAMES = frozenset(DEFAULTS)


class ConfigError(ValueError):
    """Invalid user configuration."""


@dataclass(frozen=True)
class ValidatedConfig:
    n_layers: int
    alpha: float
    f_int: float
    f_irr: float
    gravity: float
    p_bottom_bar: float
    p_top_bar: float
    p_bottom: float
    p_top: float
    x_he: float
    seed: str
    opacity_model: str
    procedure: str
    output_dir: str
    out_name: str
    overwrite: bool
    max_steps_live_polish: int
    max_steps_continuation: int
    max_recovery_cycles: int
    dt_accuracy_s: float
    dt_hold_init_s: float
    continuation_dt_accuracy_s: float
    prescribed_dt_s: float | None
    max_steps_adaptive_only: int
    write_profiles: bool
    write_convergence: bool
    write_result_json: bool
    write_status: bool
    write_figure: bool
    figure_dpi: int
    plot_temperature: bool
    plot_fluxes: bool
    plot_gradients: bool
    plot_convergence: bool
    gate: float
    envelope_status: str
    envelope_warnings: tuple[str, ...]
    config_path: str
    config_checksum_sha256: str

    @property
    def resolved_output_dir(self) -> str:
        base = self.output_dir.rstrip("/")
        if self.out_name.strip():
            return f"{base}/{self.out_name.strip()}"
        return base

    def controls(self) -> ProductionControls:
        return ProductionControls(
            max_steps_live_polish=self.max_steps_live_polish,
            max_steps_continuation=self.max_steps_continuation,
            max_recovery_cycles=self.max_recovery_cycles,
            dt_accuracy_s=self.dt_accuracy_s,
            dt_hold_init_s=self.dt_hold_init_s,
            continuation_dt_accuracy_s=self.continuation_dt_accuracy_s,
            prescribed_dt_s=self.prescribed_dt_s,
            max_steps_adaptive_only=self.max_steps_adaptive_only,
            gate=self.gate,
        )

    def to_snapshot(self) -> dict[str, Any]:
        thermo = production_thermo(self.x_he)
        t_ref = 1500.0
        nabla_ad_ref = _scalar_thermo(thermo.nabla_ad_at(t_ref))
        eos = "ConstantH2Thermo" if self.x_he == 0.0 else "h2_he_mixture"
        return {
            "config_path": self.config_path,
            "atmosphere": {
                "n_layers": self.n_layers,
                "p_bottom_bar": self.p_bottom_bar,
                "p_top_bar": self.p_top_bar,
                "p_bottom_Pa": self.p_bottom,
                "p_top_Pa": self.p_top,
                "gravity_m_s2": self.gravity,
                "x_he": self.x_he,
                "eos": eos,
                "nabla_ad_at_1500K": nabla_ad_ref,
                "cp_at_1500K": _scalar_thermo(thermo.specific_heat(t_ref)),
            },
            "boundary_fluxes": {
                "f_int_W_m2": self.f_int,
                "f_irr_W_m2": self.f_irr,
            },
            "convection_opacity": {
                "alpha": self.alpha,
                "seed": self.seed,
                "opacity_model": self.opacity_model,
                "opacity_description": (
                    "Analytic grey κ = κ0 (P/P0)^a with a=0.5, b=0, "
                    "tau_total=100 (AnalyticOpacityRCESpec defaults)."
                ),
            },
            "solver": {
                "procedure": self.procedure,
                "max_steps_live_polish": self.max_steps_live_polish,
                "max_steps_continuation": self.max_steps_continuation,
                "max_recovery_cycles": self.max_recovery_cycles,
                "dt_accuracy_s": self.dt_accuracy_s,
                "dt_hold_init_s": self.dt_hold_init_s,
                "continuation_dt_accuracy_s": self.continuation_dt_accuracy_s,
                "prescribed_dt_s": self.prescribed_dt_s,
                "max_steps_adaptive_only": self.max_steps_adaptive_only,
                "gate": self.gate,
                "gate_note": (
                    "Frozen physical gate PHYSICAL_GATE=1e-3; "
                    "not set in the user .py case file."
                ),
            },
            "output": {
                "output_dir": self.output_dir,
                "out_name": self.out_name,
                "resolved_output_dir": self.resolved_output_dir,
                "overwrite": self.overwrite,
                "write_profiles": self.write_profiles,
                "write_convergence": self.write_convergence,
                "write_result_json": self.write_result_json,
                "write_status": self.write_status,
                "write_figure": self.write_figure,
                "figure_dpi": self.figure_dpi,
                "plot_temperature": self.plot_temperature,
                "plot_fluxes": self.plot_fluxes,
                "plot_gradients": self.plot_gradients,
                "plot_convergence": self.plot_convergence,
            },
            "flux_sign_convention": FLUX_SIGN_CONVENTION,
            "validation_envelope": self.envelope_status,
            "validation_envelope_warnings": list(self.envelope_warnings),
            "config_checksum_sha256": self.config_checksum_sha256,
        }


def _finite_positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite number") from exc
    if not (v == v) or abs(v) == float("inf"):
        raise ConfigError(f"{name} must be finite")
    if allow_zero:
        if v < 0.0:
            raise ConfigError(f"{name} must be >= 0")
    elif v <= 0.0:
        raise ConfigError(f"{name} must be > 0")
    return v


def _bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{name} must be True or False")


def _load_py_namespace(path: Path) -> dict[str, Any]:
    if path.suffix != ".py":
        raise ConfigError(f"config must be a .py file, got {path.name!r}")
    spec = importlib.util.spec_from_file_location("user_rce_cfg", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot load config file {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for forbidden in FORBIDDEN_USER_VARS:
        if hasattr(mod, forbidden):
            raise ConfigError(
                f"{forbidden!r} is not user-configurable "
                f"(physical gate is frozen at {PHYSICAL_GATE:g})"
            )
    return {name: getattr(mod, name) for name in USER_VAR_NAMES if hasattr(mod, name)}


def validate_user_cfg(raw: dict[str, Any], *, config_path: str = "") -> ValidatedConfig:
    unknown = set(raw) - USER_VAR_NAMES
    if unknown:
        raise ConfigError(f"unknown config variables: {sorted(unknown)}")

    merged = {**DEFAULTS, **raw}

    n_layers = int(merged["n_layers"])
    if n_layers < MIN_N_LAYERS or n_layers > MAX_N_LAYERS:
        raise ConfigError(
            f"n_layers={n_layers} out of range; require {MIN_N_LAYERS} <= n_layers <= {MAX_N_LAYERS}"
        )
    if n_layers not in ALLOWED_N_LAYERS:
        pass  # allowed; validation_envelope will warn if outside nested family

    alpha = _finite_positive("alpha", merged["alpha"])
    f_int = _finite_positive("f_int", merged["f_int"])
    f_irr = _finite_positive("f_irr", merged["f_irr"], allow_zero=True)
    gravity = _finite_positive("gravity", merged["gravity"])
    p_bottom_bar = _finite_positive("p_bottom", merged["p_bottom"])
    p_top_bar = _finite_positive("p_top", merged["p_top"])
    if not (p_bottom_bar > p_top_bar > 0.0):
        raise ConfigError("require p_bottom > p_top > 0 (pressures in bar)")
    p_bottom = p_bottom_bar * BAR_TO_PA
    p_top = p_top_bar * BAR_TO_PA

    try:
        x_he = float(merged["x_he"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("x_he must be a number in [0, 1]") from exc
    if not (0.0 <= x_he <= 1.0) or not np.isfinite(x_he):
        raise ConfigError("x_he must be a finite number in [0, 1]")

    seed = str(merged["seed"])
    if seed not in ALLOWED_SEEDS:
        raise ConfigError(
            f"seed must be one of {sorted(ALLOWED_SEEDS)} "
            "(isothermal omitted until production-tested)"
        )

    opacity_model = str(merged["opacity_model"])
    if opacity_model != OPACITY_MODEL:
        raise ConfigError(f"opacity_model must be {OPACITY_MODEL!r} in v1")

    procedure = str(merged["procedure"])
    if procedure not in ALLOWED_PROCEDURES:
        raise ConfigError(f"procedure must be one of {sorted(ALLOWED_PROCEDURES)}")

    output_dir = str(merged["output_dir"]).strip()
    if not output_dir:
        raise ConfigError("output_dir must be non-empty")
    out_name = str(merged["out_name"] or "")

    def _int_nonneg(name: str) -> int:
        v = int(merged[name])
        if v < 0:
            raise ConfigError(f"{name} must be >= 0")
        return v

    max_steps_live = _int_nonneg("max_steps_live_polish")
    max_steps_cont = _int_nonneg("max_steps_continuation")
    max_recovery = _int_nonneg("max_recovery_cycles")
    max_adaptive = _int_nonneg("max_steps_adaptive_only")
    if max_steps_live < 1 and procedure == "production":
        raise ConfigError("max_steps_live_polish must be >= 1 for production")

    dt_accuracy = _finite_positive("dt_accuracy_s", merged["dt_accuracy_s"])
    dt_hold = _finite_positive("dt_hold_init_s", merged["dt_hold_init_s"])
    cont_dt = _finite_positive(
        "continuation_dt_accuracy_s", merged["continuation_dt_accuracy_s"]
    )
    presc = merged["prescribed_dt_s"]
    if presc is not None:
        presc = _finite_positive("prescribed_dt_s", presc)

    figure_dpi = int(merged["figure_dpi"])
    if figure_dpi < 50:
        raise ConfigError("figure_dpi must be >= 50")

    write_figure = _bool("write_figure", merged["write_figure"])
    plot_temperature = _bool("plot_temperature", merged["plot_temperature"])
    plot_fluxes = _bool("plot_fluxes", merged["plot_fluxes"])
    plot_gradients = _bool("plot_gradients", merged["plot_gradients"])
    plot_convergence = _bool("plot_convergence", merged["plot_convergence"])
    if write_figure and not any(
        (plot_temperature, plot_fluxes, plot_gradients, plot_convergence)
    ):
        raise ConfigError(
            "write_figure=True but all plot_* panels are False; "
            "enable at least one panel or set write_figure=False"
        )

    env_status, env_warn = validation_envelope(
        n_layers=n_layers,
        alpha=alpha,
        f_int=f_int,
        f_irr=f_irr,
        gravity=gravity,
        p_bottom=p_bottom,
        p_top=p_top,
        composition="constant_h2",
        opacity_model=opacity_model,
        x_he=x_he,
    )

    canon = {k: merged[k] for k in sorted(USER_VAR_NAMES)}
    digest = hashlib.sha256(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return ValidatedConfig(
        n_layers=n_layers,
        alpha=alpha,
        f_int=f_int,
        f_irr=f_irr,
        gravity=gravity,
        p_bottom_bar=p_bottom_bar,
        p_top_bar=p_top_bar,
        p_bottom=p_bottom,
        p_top=p_top,
        x_he=x_he,
        seed=seed,
        opacity_model=opacity_model,
        procedure=procedure,
        output_dir=output_dir,
        out_name=out_name,
        overwrite=_bool("overwrite", merged["overwrite"]),
        max_steps_live_polish=max_steps_live,
        max_steps_continuation=max_steps_cont,
        max_recovery_cycles=max_recovery,
        dt_accuracy_s=dt_accuracy,
        dt_hold_init_s=dt_hold,
        continuation_dt_accuracy_s=cont_dt,
        prescribed_dt_s=presc,
        max_steps_adaptive_only=max_adaptive,
        write_profiles=_bool("write_profiles", merged["write_profiles"]),
        write_convergence=_bool("write_convergence", merged["write_convergence"]),
        write_result_json=_bool("write_result_json", merged["write_result_json"]),
        write_status=_bool("write_status", merged["write_status"]),
        write_figure=write_figure,
        figure_dpi=figure_dpi,
        plot_temperature=plot_temperature,
        plot_fluxes=plot_fluxes,
        plot_gradients=plot_gradients,
        plot_convergence=plot_convergence,
        gate=PHYSICAL_GATE,
        envelope_status=env_status,
        envelope_warnings=tuple(env_warn),
        config_path=config_path,
        config_checksum_sha256=digest,
    )


def load_and_validate(path: Path) -> ValidatedConfig:
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = _load_py_namespace(path)
    return validate_user_cfg(raw, config_path=str(path))


def write_example_config(path: Path) -> None:
    template = Path(__file__).resolve().parent / "cfg_demo.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, path)


def copy_config_to_run_dir(cfg_path: Path, out_dir: Path) -> Path:
    dest = out_dir / "input_cfg.py"
    shutil.copy2(cfg_path, dest)
    return dest

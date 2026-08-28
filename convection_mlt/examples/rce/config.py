"""Validated user-facing RCE configuration (units documented; frozen gate)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from convection_mlt.production_rce import (
    DEFAULT_ALPHA,
    DEFAULT_F_INT,
    DEFAULT_F_IRR,
    DEFAULT_GRAVITY,
    DEFAULT_P_BOTTOM,
    DEFAULT_P_TOP,
    PHYSICAL_GATE,
    ProductionControls,
    validation_envelope,
)
from convection_mlt.thermodynamics import ConstantH2Thermo

ALLOWED_N_LAYERS = frozenset({96, 192, 384})  # 48 rejected: not in demonstrated sequence
ALLOWED_SEEDS = frozenset({"radiative_convective", "radiative_equilibrium"})
ALLOWED_PROCEDURES = frozenset({"production", "adaptive_only"})
ALLOWED_COMPOSITION = frozenset({"constant_h2"})
OPACITY_MODEL = "analytic_grey_powerlaw"

FLUX_SIGN_CONVENTION = {
    "f_int_W_m2": (
        "Net upward internal flux imposed at the bottom boundary [W m^-2]."
    ),
    "f_irr_W_m2": (
        "Downward irradiation at the top boundary [W m^-2]; 0 in the default demo."
    ),
    "profile_fluxes": (
        "Positive flux is upward net energy transport; "
        "F_total = F_rad + F_conv on interfaces."
    ),
}

UNITS = {
    "n_layers": "count",
    "alpha": "dimensionless mixing-length parameter",
    "f_int_W_m2": "W m^-2",
    "f_irr_W_m2": "W m^-2",
    "gravity_m_s2": "m s^-2",
    "p_bottom_Pa": "Pa",
    "p_top_Pa": "Pa",
    "dt_accuracy_s": "s (pseudo-time accuracy ceiling unless prescribed_dt_s set)",
    "dt_hold_init_s": "s",
    "prescribed_dt_s": "s (physical Δt when set; else histories are pseudo-time)",
}


class ConfigError(ValueError):
    """Invalid user configuration."""


@dataclass(frozen=True)
class ValidatedConfig:
    n_layers: int
    alpha: float
    f_int_W_m2: float
    f_irr_W_m2: float
    gravity_m_s2: float
    p_bottom_Pa: float
    p_top_Pa: float
    composition: str
    seed: str
    opacity_model: str
    procedure: str
    output_dir: str
    max_steps_live_polish: int
    max_steps_continuation: int
    max_recovery_cycles: int
    dt_accuracy_s: float
    dt_hold_init_s: float
    continuation_dt_accuracy_s: float
    prescribed_dt_s: float | None
    max_steps_adaptive_only: int
    gate: float
    envelope_status: str
    envelope_warnings: tuple[str, ...]
    nabla_ad: float
    config_checksum_sha256: str

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
        thermo = ConstantH2Thermo()
        return {
            "physics": {
                "n_layers": self.n_layers,
                "alpha": self.alpha,
                "f_int_W_m2": self.f_int_W_m2,
                "f_irr_W_m2": self.f_irr_W_m2,
                "gravity_m_s2": self.gravity_m_s2,
                "p_bottom_Pa": self.p_bottom_Pa,
                "p_top_Pa": self.p_top_Pa,
                "composition": self.composition,
                "seed": self.seed,
                "opacity_model": self.opacity_model,
                "opacity_description": (
                    "Analytic grey κ = κ0 (P/P0)^a with a=0.5, b=0, "
                    "tau_total=100 (AnalyticOpacityRCESpec defaults)."
                ),
                "nabla_ad": self.nabla_ad,
                "cp": float(thermo.cp),
            },
            "solve": {
                "procedure": self.procedure,
                "output_dir": self.output_dir,
                "gate": self.gate,
                "gate_note": (
                    "Frozen physical gate PHYSICAL_GATE=1e-3; "
                    "users may not relax it above 1e-3."
                ),
            },
            "advanced": {
                "max_steps_live_polish": self.max_steps_live_polish,
                "max_steps_continuation": self.max_steps_continuation,
                "max_recovery_cycles": self.max_recovery_cycles,
                "dt_accuracy_s": self.dt_accuracy_s,
                "dt_hold_init_s": self.dt_hold_init_s,
                "continuation_dt_accuracy_s": self.continuation_dt_accuracy_s,
                "prescribed_dt_s": self.prescribed_dt_s,
                "max_steps_adaptive_only": self.max_steps_adaptive_only,
                "prescribed_dt_applies_to": (
                    "live_polish and continuation phases under procedure=production; "
                    "entire run under procedure=adaptive_only"
                ),
            },
            "units": UNITS,
            "flux_sign_convention": FLUX_SIGN_CONVENTION,
            "validation_envelope": self.envelope_status,
            "validation_envelope_warnings": list(self.envelope_warnings),
            "config_checksum_sha256": self.config_checksum_sha256,
        }


def default_example_dict() -> dict[str, Any]:
    return {
        "physics": {
            "n_layers": 96,
            "alpha": 1.0,
            "f_int_W_m2": 300.0,
            "f_irr_W_m2": 0.0,
            "gravity_m_s2": 15.0,
            "p_bottom_Pa": 1.0e6,
            "p_top_Pa": 1.0,
            "composition": "constant_h2",
            "seed": "radiative_convective",
            "opacity_model": OPACITY_MODEL,
        },
        "solve": {
            "procedure": "production",
            "output_dir": "examples/rce/runs/demo_n96_alpha1",
        },
        "advanced": {
            "max_steps_live_polish": 200,
            "max_steps_continuation": 500,
            "max_recovery_cycles": 2,
            "dt_accuracy_s": 50000.0,
            "dt_hold_init_s": 18415.0,
            "continuation_dt_accuracy_s": 2500.0,
            "prescribed_dt_s": None,
            "max_steps_adaptive_only": 20000,
        },
    }


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be a JSON object")
    return data


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


def validate_config(raw: dict[str, Any]) -> ValidatedConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    allowed_top = {"physics", "solve", "advanced", "gate"}
    unknown = set(raw) - allowed_top
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")

    # Legacy / forbidden gate loosening
    legacy_gate = None
    if "gate" in raw:
        legacy_gate = _finite_positive("gate", raw["gate"])
    physics = _require_mapping(raw.get("physics", {}), "physics")
    solve = _require_mapping(raw.get("solve", {}), "solve")
    advanced = _require_mapping(raw.get("advanced", {}), "advanced")

    if "gate" in solve:
        legacy_gate = _finite_positive("solve.gate", solve["gate"])
    if "adaptive_only" in advanced:
        raise ConfigError(
            "advanced.adaptive_only is removed; set solve.procedure to "
            "'production' or 'adaptive_only'"
        )

    phys_allowed = {
        "n_layers",
        "alpha",
        "f_int_W_m2",
        "f_irr_W_m2",
        "gravity_m_s2",
        "p_bottom_Pa",
        "p_top_Pa",
        "composition",
        "seed",
        "opacity_model",
    }
    unk_p = set(physics) - phys_allowed
    if unk_p:
        raise ConfigError(f"unknown physics keys: {sorted(unk_p)}")
    solve_allowed = {"procedure", "output_dir", "gate"}
    unk_s = set(solve) - solve_allowed
    if unk_s:
        raise ConfigError(f"unknown solve keys: {sorted(unk_s)}")
    adv_allowed = {
        "max_steps_live_polish",
        "max_steps_continuation",
        "max_recovery_cycles",
        "dt_accuracy_s",
        "dt_hold_init_s",
        "continuation_dt_accuracy_s",
        "prescribed_dt_s",
        "max_steps_adaptive_only",
    }
    unk_a = set(advanced) - adv_allowed
    if unk_a:
        raise ConfigError(f"unknown advanced keys: {sorted(unk_a)}")

    n_layers = int(physics.get("n_layers", 96))
    if n_layers not in ALLOWED_N_LAYERS:
        raise ConfigError(
            f"n_layers={n_layers} not allowed; choose one of {sorted(ALLOWED_N_LAYERS)} "
            "(N=48 is not part of the demonstrated spatial sequence)"
        )
    alpha = _finite_positive("alpha", physics.get("alpha", DEFAULT_ALPHA))
    f_int = _finite_positive("f_int_W_m2", physics.get("f_int_W_m2", DEFAULT_F_INT))
    f_irr = _finite_positive(
        "f_irr_W_m2", physics.get("f_irr_W_m2", DEFAULT_F_IRR), allow_zero=True
    )
    gravity = _finite_positive("gravity_m_s2", physics.get("gravity_m_s2", DEFAULT_GRAVITY))
    p_bottom = _finite_positive("p_bottom_Pa", physics.get("p_bottom_Pa", DEFAULT_P_BOTTOM))
    p_top = _finite_positive("p_top_Pa", physics.get("p_top_Pa", DEFAULT_P_TOP))
    if not (p_bottom > p_top > 0.0):
        raise ConfigError("require p_bottom_Pa > p_top_Pa > 0")

    composition = str(physics.get("composition", "constant_h2"))
    if composition not in ALLOWED_COMPOSITION:
        raise ConfigError(f"composition must be one of {sorted(ALLOWED_COMPOSITION)}")
    seed = str(physics.get("seed", "radiative_convective"))
    if seed not in ALLOWED_SEEDS:
        raise ConfigError(
            f"seed must be one of {sorted(ALLOWED_SEEDS)} "
            "(isothermal omitted until production-tested)"
        )
    opacity_model = str(physics.get("opacity_model", OPACITY_MODEL))
    if opacity_model != OPACITY_MODEL:
        raise ConfigError(f"opacity_model must be {OPACITY_MODEL!r} in v1")

    procedure = str(solve.get("procedure", "production"))
    if procedure not in ALLOWED_PROCEDURES:
        raise ConfigError(f"solve.procedure must be one of {sorted(ALLOWED_PROCEDURES)}")
    output_dir = str(solve.get("output_dir", "examples/rce/runs/demo_n96_alpha1"))
    if not output_dir.strip():
        raise ConfigError("solve.output_dir must be non-empty")

    gate = PHYSICAL_GATE
    if legacy_gate is not None:
        if legacy_gate > PHYSICAL_GATE + 1.0e-15:
            raise ConfigError(
                f"gate={legacy_gate} exceeds frozen PHYSICAL_GATE={PHYSICAL_GATE}; "
                "users may not relax the physical gate"
            )
        gate = float(legacy_gate)

    def _int_pos(name: str, default: int) -> int:
        v = int(advanced.get(name, default))
        if v < 0:
            raise ConfigError(f"{name} must be >= 0")
        return v

    max_steps_live = _int_pos("max_steps_live_polish", 200)
    max_steps_cont = _int_pos("max_steps_continuation", 500)
    max_recovery = _int_pos("max_recovery_cycles", 2)
    max_adaptive = _int_pos("max_steps_adaptive_only", 20000)
    if max_steps_live < 1 and procedure == "production":
        raise ConfigError("max_steps_live_polish must be >= 1 for production")
    dt_accuracy = _finite_positive("dt_accuracy_s", advanced.get("dt_accuracy_s", 50000.0))
    dt_hold = _finite_positive("dt_hold_init_s", advanced.get("dt_hold_init_s", 18415.0))
    cont_dt = _finite_positive(
        "continuation_dt_accuracy_s",
        advanced.get("continuation_dt_accuracy_s", 2500.0),
    )
    presc = advanced.get("prescribed_dt_s", None)
    if presc is not None:
        presc = _finite_positive("prescribed_dt_s", presc)

    env_status, env_warn = validation_envelope(
        n_layers=n_layers,
        alpha=alpha,
        f_int=f_int,
        f_irr=f_irr,
        gravity=gravity,
        p_bottom=p_bottom,
        p_top=p_top,
        composition=composition,
        opacity_model=opacity_model,
    )
    thermo = ConstantH2Thermo()
    # checksum over canonical physics+solve+advanced (excluding derived fields)
    canon = {
        "physics": {
            "n_layers": n_layers,
            "alpha": alpha,
            "f_int_W_m2": f_int,
            "f_irr_W_m2": f_irr,
            "gravity_m_s2": gravity,
            "p_bottom_Pa": p_bottom,
            "p_top_Pa": p_top,
            "composition": composition,
            "seed": seed,
            "opacity_model": opacity_model,
        },
        "solve": {"procedure": procedure, "output_dir": output_dir, "gate": gate},
        "advanced": {
            "max_steps_live_polish": max_steps_live,
            "max_steps_continuation": max_steps_cont,
            "max_recovery_cycles": max_recovery,
            "dt_accuracy_s": dt_accuracy,
            "dt_hold_init_s": dt_hold,
            "continuation_dt_accuracy_s": cont_dt,
            "prescribed_dt_s": presc,
            "max_steps_adaptive_only": max_adaptive,
        },
    }
    digest = hashlib.sha256(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return ValidatedConfig(
        n_layers=n_layers,
        alpha=alpha,
        f_int_W_m2=f_int,
        f_irr_W_m2=f_irr,
        gravity_m_s2=gravity,
        p_bottom_Pa=p_bottom,
        p_top_Pa=p_top,
        composition=composition,
        seed=seed,
        opacity_model=opacity_model,
        procedure=procedure,
        output_dir=output_dir,
        max_steps_live_polish=max_steps_live,
        max_steps_continuation=max_steps_cont,
        max_recovery_cycles=max_recovery,
        dt_accuracy_s=dt_accuracy,
        dt_hold_init_s=dt_hold,
        continuation_dt_accuracy_s=cont_dt,
        prescribed_dt_s=presc,
        max_steps_adaptive_only=max_adaptive,
        gate=gate,
        envelope_status=env_status,
        envelope_warnings=tuple(env_warn),
        nabla_ad=float(thermo.nabla_ad),
        config_checksum_sha256=digest,
    )


def load_and_validate(path: Path) -> ValidatedConfig:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON: {exc}") from exc
    return validate_config(raw)


def write_example_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(default_example_dict(), indent=2) + "\n")

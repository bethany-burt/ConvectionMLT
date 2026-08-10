"""Configuration for the R0 calorically-perfect dry-H2 reference model."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float = 15.0
    alpha: float = 1.0
    closure_prefactor: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.gravity) or self.gravity <= 0.0:
            raise ValueError("gravity must be finite and positive")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and nonnegative")
        if (
            not math.isfinite(self.closure_prefactor)
            or self.closure_prefactor <= 0.0
        ):
            raise ValueError("closure_prefactor must be finite and positive")


@dataclass(frozen=True)
class SolverConfig:
    epsilon_gradient: float = 1.0e-8
    epsilon_temperature: float = 1.0e-3
    c_diff: float = 0.2
    c_active: float = 10.0
    c_cross: float = 1.0
    f_back: float = 0.5
    max_rejections: int = 50
    dt_min: float = 1.0e-12
    max_steps: int = 1_000_000
    theta_rms_tolerance: float = 1.0e-8
    temperature_rms_tolerance: float = 1.0e-8
    temperature_max_tolerance: float = 1.0e-7
    tendency_tolerance: float = 1.0e-12
    flux_tolerance: float = 5.0e-3
    enthalpy_drift_tolerance: float = 1.0e-10
    transfer_merge_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        positive = {
            "epsilon_gradient": self.epsilon_gradient,
            "epsilon_temperature": self.epsilon_temperature,
            "c_diff": self.c_diff,
            "c_active": self.c_active,
            "c_cross": self.c_cross,
            "dt_min": self.dt_min,
            "theta_rms_tolerance": self.theta_rms_tolerance,
            "temperature_rms_tolerance": self.temperature_rms_tolerance,
            "temperature_max_tolerance": self.temperature_max_tolerance,
            "tendency_tolerance": self.tendency_tolerance,
            "flux_tolerance": self.flux_tolerance,
            "enthalpy_drift_tolerance": self.enthalpy_drift_tolerance,
            "transfer_merge_tolerance": self.transfer_merge_tolerance,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.f_back) or not 0.0 < self.f_back < 1.0:
            raise ValueError("f_back must be finite and between zero and one")
        if self.max_rejections < 0:
            raise ValueError("max_rejections must be nonnegative")
        if self.max_steps < 0:
            raise ValueError("max_steps must be nonnegative")

    def as_metadata(self) -> dict[str, float | int]:
        return asdict(self)

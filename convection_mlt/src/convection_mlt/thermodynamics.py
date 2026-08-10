"""Calorically perfect ideal molecular-hydrogen benchmark thermodynamics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validate import positive, temperatures

R_UNIVERSAL = 8.31446261815324  # J mol^-1 K^-1
H2_MOLAR_MASS = 2.01588e-3  # kg mol^-1


@dataclass(frozen=True)
class IdealH2:
    """Constant-cp H2 used only as the controlled R0 validation medium."""

    molar_mass: float = H2_MOLAR_MASS

    def __post_init__(self) -> None:
        positive("molar_mass", self.molar_mass)

    @property
    def gas_constant(self) -> float:
        return R_UNIVERSAL / self.molar_mass

    @property
    def cp(self) -> float:
        return 3.5 * self.gas_constant

    @property
    def nabla_ad(self) -> float:
        return self.gas_constant / self.cp

    def density(self, pressure: ArrayLike, temperature: ArrayLike) -> NDArray[np.float64]:
        pressure_array = np.asarray(pressure, dtype=float)
        temperature_array = temperatures(temperature)
        if np.any(~np.isfinite(pressure_array)) or np.any(pressure_array <= 0.0):
            raise ValueError("pressure must be finite and positive")
        return pressure_array / (self.gas_constant * temperature_array)

    def scale_height(self, temperature: ArrayLike, gravity: float) -> NDArray[np.float64]:
        positive("gravity", gravity)
        return self.gas_constant * temperatures(temperature) / gravity

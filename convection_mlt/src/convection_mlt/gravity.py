"""Gravity laws for Stage 2 hydrostatics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validate import positive


@runtime_checkable
class GravityLaw(Protocol):
    def gravity(self, z: ArrayLike) -> NDArray[np.float64]: ...
    def as_metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ConstantGravity:
    g0: float

    def __post_init__(self) -> None:
        positive("g0", self.g0)

    def gravity(self, z: ArrayLike) -> NDArray[np.float64]:
        height = np.asarray(z, dtype=float)
        return np.full(height.shape, self.g0, dtype=float)

    def as_metadata(self) -> dict[str, Any]:
        return {"mode": "constant", "g0": self.g0}


@dataclass(frozen=True)
class InverseSquareGravity:
    """g(z) = GM / (R_p + z)^2 with GM = g0 * R_p^2 for controlled comparisons."""

    g0: float
    planet_radius: float
    gravitational_parameter: float | None = None

    def __post_init__(self) -> None:
        positive("g0", self.g0)
        positive("planet_radius", self.planet_radius)
        if self.gravitational_parameter is not None:
            positive("gravitational_parameter", self.gravitational_parameter)

    @property
    def gm(self) -> float:
        if self.gravitational_parameter is None:
            return self.g0 * self.planet_radius**2
        return float(self.gravitational_parameter)

    def gravity(self, z: ArrayLike) -> NDArray[np.float64]:
        height = np.asarray(z, dtype=float)
        if np.any(~np.isfinite(height)):
            raise ValueError("z must be finite")
        if np.any(height < -self.planet_radius):
            raise ValueError("z must satisfy R_p + z > 0")
        radius = self.planet_radius + height
        return self.gm / radius**2

    def as_metadata(self) -> dict[str, Any]:
        return {
            "mode": "inverse_square",
            "g0": self.g0,
            "planet_radius": self.planet_radius,
            "gravitational_parameter": self.gm,
        }

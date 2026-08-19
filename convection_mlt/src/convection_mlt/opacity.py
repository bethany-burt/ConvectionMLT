"""Prescribed opacity providers for Stage 3 radiative transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class PrescribedOpacity(Protocol):
    """Return absorption opacity κ [m² kg⁻¹] with shape (n_band, n_layer)."""

    @property
    def n_band(self) -> int: ...

    @property
    def band_weights(self) -> NDArray[np.float64]: ...

    def evaluate(
        self,
        temperature: NDArray[np.float64],
        pressure: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class ConstantGreyOpacity:
    """Constant grey opacity κ₀ across all layers."""

    kappa0: float

    @property
    def n_band(self) -> int:
        return 1

    @property
    def band_weights(self) -> NDArray[np.float64]:
        return np.array([1.0])

    def evaluate(
        self,
        temperature: NDArray[np.float64],
        pressure: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.kappa0 < 0 or not np.isfinite(self.kappa0):
            raise ValueError("kappa0 must be finite and nonnegative")
        n = temperature.shape[0]
        return np.full((1, n), self.kappa0, dtype=np.float64)


@dataclass(frozen=True)
class AnalyticGreyOpacity:
    """κ(T, P) = κ₀ (P / P₀)^a (T / T₀)^b."""

    kappa0: float
    P0: float
    T0: float
    a: float
    b: float

    @property
    def n_band(self) -> int:
        return 1

    @property
    def band_weights(self) -> NDArray[np.float64]:
        return np.array([1.0])

    def evaluate(
        self,
        temperature: NDArray[np.float64],
        pressure: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        kappa = self.kappa0 * (pressure / self.P0) ** self.a * (temperature / self.T0) ** self.b
        result = kappa[np.newaxis, :]
        if np.any(result < 0) or not np.all(np.isfinite(result)):
            raise ValueError("Evaluated opacity must be finite and nonnegative")
        return result


@dataclass(frozen=True)
class PrescribedBandOpacity:
    """Multiband opacity with prescribed per-band κ arrays and weights."""

    kappa_bands: NDArray[np.float64]   # (n_band, n_layer)
    weights: NDArray[np.float64]       # (n_band,)

    def __post_init__(self) -> None:
        if self.kappa_bands.ndim != 2:
            raise ValueError("kappa_bands must be 2-d (n_band, n_layer)")
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1-d")
        if self.weights.shape[0] != self.kappa_bands.shape[0]:
            raise ValueError("weights length must match kappa_bands first axis")
        _validate_band_weights(self.weights)
        if np.any(self.kappa_bands < 0) or not np.all(np.isfinite(self.kappa_bands)):
            raise ValueError("kappa_bands must be finite and nonnegative")

    @property
    def n_band(self) -> int:
        return int(self.kappa_bands.shape[0])

    @property
    def band_weights(self) -> NDArray[np.float64]:
        return self.weights.copy()

    def evaluate(
        self,
        temperature: NDArray[np.float64],
        pressure: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.kappa_bands.copy()


def _validate_band_weights(w: NDArray[np.float64]) -> None:
    if w.ndim != 1 or w.size == 0:
        raise ValueError("band_weights must be a non-empty 1-d array")
    if not np.all(np.isfinite(w)):
        raise ValueError("band_weights must be finite")
    if np.any(w < 0):
        raise ValueError("band_weights must be >= 0")
    if not np.any(w > 0):
        raise ValueError("At least one band weight must be positive")
    if abs(np.sum(w) - 1.0) > 1e-15:
        raise ValueError(f"band_weights must sum to 1, got {np.sum(w)}")

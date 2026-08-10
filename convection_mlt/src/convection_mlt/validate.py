"""Input validation shared by all R0 kernels."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def finite_1d(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def pressure_edges(values: ArrayLike) -> NDArray[np.float64]:
    edges = finite_1d("pressure_edges", values)
    if edges.size < 2:
        raise ValueError("pressure_edges must contain at least two edges")
    if np.any(edges <= 0.0):
        raise ValueError("pressure_edges must be positive")
    if not np.all(np.diff(edges) < 0.0):
        raise ValueError("pressure_edges must be strictly decreasing bottom-to-top")
    return edges


def temperatures(values: ArrayLike, expected_size: int | None = None) -> NDArray[np.float64]:
    temperature = finite_1d("temperature", values)
    if expected_size is not None and temperature.size != expected_size:
        raise ValueError(f"temperature must have length {expected_size}")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature must be positive")
    return temperature

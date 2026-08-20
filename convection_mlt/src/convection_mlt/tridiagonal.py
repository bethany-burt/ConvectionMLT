"""Thomas algorithm for tridiagonal systems (Stage 3/4 helper)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ThomasPivotError(ValueError):
    """Tridiagonal factorization hit a nonfinite or near-zero pivot."""


def thomas_solve(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve A x = b where A is tridiagonal.

    Parameters
    ----------
    lower : (n-1,) sub-diagonal
    diag  : (n,) main diagonal
    upper : (n-1,) super-diagonal
    rhs   : (n,) right-hand side

    Returns
    -------
    x : (n,) solution
    """
    n = diag.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    d = diag.copy()
    b = rhs.copy()

    for i in range(1, n):
        m = lower[i - 1] / d[i - 1]
        d[i] -= m * upper[i - 1]
        b[i] -= m * b[i - 1]

    x = np.empty(n, dtype=np.float64)
    x[n - 1] = b[n - 1] / d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - upper[i] * x[i + 1]) / d[i]

    return x


def thomas_solve_checked(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
    *,
    pivot_floor: float = 1.0e-30,
) -> NDArray[np.float64]:
    """Thomas solve with nonfinite / near-zero pivot rejection."""
    n = diag.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    d = diag.copy()
    b = rhs.copy()
    if not np.isfinite(d[0]) or abs(d[0]) < pivot_floor:
        raise ThomasPivotError(f"thomas pivot 0 nonfinite or below floor ({d[0]!r})")

    for i in range(1, n):
        if not np.isfinite(d[i - 1]) or abs(d[i - 1]) < pivot_floor:
            raise ThomasPivotError(
                f"thomas pivot {i - 1} nonfinite or below floor ({d[i - 1]!r})"
            )
        m = lower[i - 1] / d[i - 1]
        d[i] -= m * upper[i - 1]
        b[i] -= m * b[i - 1]
        if not np.isfinite(d[i]) or abs(d[i]) < pivot_floor:
            raise ThomasPivotError(f"thomas pivot {i} nonfinite or below floor ({d[i]!r})")

    x = np.empty(n, dtype=np.float64)
    x[n - 1] = b[n - 1] / d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - upper[i] * x[i + 1]) / d[i]
    if not np.all(np.isfinite(x)):
        raise ThomasPivotError("thomas solution is nonfinite")
    return x

"""Thomas algorithm for tridiagonal systems (internal Stage 3 helper)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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

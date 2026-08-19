"""Independent hydrostatic references for Figure 06.

Shares EOS, gravity law, and prescribed T(P) with the model, but does **not**
call ``reconstruct_hydrostatic`` or its layer-stepping discretization.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from convection_mlt.gravity import ConstantGravity, GravityLaw, InverseSquareGravity
from convection_mlt.hydrostatics import (
    isothermal_constant_g_height,
    isothermal_inverse_square_height,
)

# Dormand–Prince RK45 coefficients.
_A = (
    (0.0,),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
)
_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0)
_B5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0)
_B4 = (
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
)


def rk45_adaptive(
    fun: Callable[[float, float], float],
    x0: float,
    y0: float,
    x1: float,
    *,
    rtol: float,
    atol: float,
    max_step: float,
    samples: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, float | int | str]]:
    """Integrate dy/dx = fun(x, y) and interpolate the solution at ``samples``.

    ``samples`` must be monotonic in the same direction as ``x1 - x0``.
    """
    direction = 1.0 if x1 >= x0 else -1.0
    h = direction * min(abs(max_step), abs(x1 - x0))
    x = float(x0)
    y = float(y0)
    n_accept = 0
    n_reject = 0
    n_eval = 0
    min_h = abs(h)
    xs = [x]
    ys = [y]
    safety = 0.9
    while (x - x1) * direction < 0.0:
        h = direction * min(abs(h), abs(x1 - x), abs(max_step))
        k = np.empty(7, dtype=float)
        k[0] = fun(x, y)
        n_eval += 1
        for i in range(1, 6):
            yi = y + h * sum(_A[i][j] * k[j] for j in range(i))
            k[i] = fun(x + _C[i] * h, yi)
            n_eval += 1
        y5 = y + h * sum(_B5[j] * k[j] for j in range(6))
        k[6] = fun(x + h, y5)
        n_eval += 1
        y4 = y + h * sum(_B4[j] * k[j] for j in range(7))
        err = abs(y5 - y4)
        scale = atol + rtol * max(abs(y), abs(y5))
        if scale <= 0.0:
            scale = atol
        norm = err / scale
        if norm <= 1.0 or abs(h) <= 1.0e-16 * max(1.0, abs(x)):
            x = x + h
            y = float(y5)
            xs.append(x)
            ys.append(y)
            n_accept += 1
            min_h = min(min_h, abs(h))
            if norm == 0.0:
                factor = 5.0
            else:
                factor = min(5.0, max(0.2, safety * norm ** (-0.2)))
            h = direction * min(abs(max_step), abs(h) * factor)
        else:
            n_reject += 1
            factor = max(0.2, safety * norm ** (-0.25))
            h = direction * max(abs(h) * factor, 1.0e-16 * max(1.0, abs(x)))

    sample_x = np.asarray(samples, dtype=float)
    y_out = np.interp(sample_x, xs, ys) if direction > 0 else np.interp(sample_x, xs[::-1], ys[::-1])
    stats = {
        "reference_method": "rk45_dormand_prince_adaptive_ode",
        "relative_tolerance": float(rtol),
        "absolute_tolerance": float(atol),
        "maximum_step": float(max_step),
        "n_accepted_steps": int(n_accept),
        "n_rejected_steps": int(n_reject),
        "n_function_evals": int(n_eval),
        "min_accepted_step": float(min_h),
        "independent_coordinate": "ln_pressure",
    }
    return np.asarray(y_out, dtype=float), stats


def hydrostatic_rhs(
    ln_p: float,
    z: float,
    *,
    gas_constant: float,
    temperature_of_p: Callable[[float], float],
    gravity: GravityLaw,
) -> float:
    """dz / dlnP = - R T(P) / g(z)."""
    pressure = float(np.exp(ln_p))
    temperature = float(temperature_of_p(pressure))
    g = float(gravity.gravity(np.asarray([z]))[0])
    return -gas_constant * temperature / g


def integrate_z_of_pressure(
    pressure_edges: NDArray[np.float64],
    *,
    gas_constant: float,
    temperature_of_p: Callable[[float], float],
    gravity: GravityLaw,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-10,
    max_step: float = 0.005,
) -> tuple[NDArray[np.float64], dict[str, float | int | str]]:
    ln_edges = np.log(np.asarray(pressure_edges, dtype=float))

    def fun(ln_p: float, z: float) -> float:
        return hydrostatic_rhs(
            ln_p,
            z,
            gas_constant=gas_constant,
            temperature_of_p=temperature_of_p,
            gravity=gravity,
        )

    z, stats = rk45_adaptive(
        fun,
        float(ln_edges[0]),
        0.0,
        float(ln_edges[-1]),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        samples=ln_edges,
    )
    stats["maximum_step"] = float(max_step)
    stats["maximum_step_coordinate"] = "dlnP"
    return z, stats


def analytic_isothermal_constant_g_edges(
    pressure_edges: NDArray[np.float64],
    temperature: float,
    gas_constant: float,
    g0: float,
) -> NDArray[np.float64]:
    p = np.asarray(pressure_edges, dtype=float)
    return (gas_constant * temperature / g0) * np.log(p[0] / p)


def analytic_isothermal_inverse_square_edges(
    pressure_edges: NDArray[np.float64],
    temperature: float,
    gas_constant: float,
    gravity: InverseSquareGravity,
) -> NDArray[np.float64]:
    p = np.asarray(pressure_edges, dtype=float)
    z = np.zeros_like(p)
    z[-1] = isothermal_inverse_square_height(
        float(p[0]), float(p[-1]), temperature, gas_constant, gravity, 0.0
    )
    for i, pressure in enumerate(p[1:-1], start=1):
        z[i] = isothermal_inverse_square_height(
            float(p[0]), float(pressure), temperature, gas_constant, gravity, 0.0
        )
    return z


def column_scale_height_error(
    z_var: NDArray[np.float64], z_const: NDArray[np.float64]
) -> dict[str, float]:
    z_var = np.asarray(z_var, dtype=float)
    z_const = np.asarray(z_const, dtype=float)
    scale = float(np.max(np.abs(z_const)))
    ez = float(np.max(np.abs(z_var - z_const)) / scale) if scale > 0.0 else float("nan")
    z_top_const = float(z_const[-1])
    ez_top = (
        abs(float(z_var[-1]) - z_top_const) / abs(z_top_const)
        if z_top_const != 0.0
        else float("nan")
    )
    return {"E_z": ez, "E_z_top": ez_top}


__all__ = [
    "analytic_isothermal_constant_g_edges",
    "analytic_isothermal_inverse_square_edges",
    "column_scale_height_error",
    "integrate_z_of_pressure",
    "isothermal_constant_g_height",
    "ConstantGravity",
]

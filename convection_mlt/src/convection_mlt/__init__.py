"""R0 reference kernel for closed-column dry-H2 mixing-length convection."""

from .closure import ClosureResult, mixing_length_flux
from .config import PhysicsConfig, SolverConfig
from .diagnostics import (
    ConvergenceMetrics,
    enthalpy_normalized_adiabat,
    mixing_region_labels,
    piecewise_enthalpy_reference,
    reference_enthalpy_residuals,
)
from .grid import PressureGrid, build_grid, log_pressure_edges
from .solvers import (
    IntegrationResult,
    SolverFailure,
    TerminalStatus,
    fixed_step,
    solve_adaptive,
)
from .thermodynamics import IdealH2
from .trace import IntegrationTrace, TraceLevel, make_trace

__all__ = [
    "ClosureResult",
    "ConvergenceMetrics",
    "IdealH2",
    "IntegrationResult",
    "IntegrationTrace",
    "PhysicsConfig",
    "PressureGrid",
    "SolverConfig",
    "SolverFailure",
    "TerminalStatus",
    "TraceLevel",
    "build_grid",
    "enthalpy_normalized_adiabat",
    "fixed_step",
    "log_pressure_edges",
    "make_trace",
    "mixing_region_labels",
    "mixing_length_flux",
    "piecewise_enthalpy_reference",
    "reference_enthalpy_residuals",
    "solve_adaptive",
]

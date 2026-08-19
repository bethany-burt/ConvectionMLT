"""R0/Stage-2 reference kernel for mixing-length convection."""

from .closure import ClosureResult, mixing_length_flux
from .config import PhysicsConfig, SolverConfig
from .diagnostics import (
    ConvergenceMetrics,
    enthalpy_normalized_adiabat,
    mixing_region_labels,
    numerical_isentrope,
    piecewise_enthalpy_reference,
    reference_enthalpy_residuals,
)
from .energy import column_enthalpy_per_area
from .gravity import ConstantGravity, InverseSquareGravity
from .grid import PressureGrid, build_grid, log_pressure_edges
from .hydrostatics import HydrostaticDomainError, reconstruct_hydrostatic
from .solvers import (
    IntegrationResult,
    SolverFailure,
    TerminalStatus,
    fixed_step,
    solve_adaptive,
)
from .solvers_enthalpy import solve_adaptive_enthalpy
from .thermodynamics import (
    AnalyticIdealGasThermo,
    ConstantH2Thermo,
    EnthalpyInversionError,
    IdealH2,
    MixtureThermo,
    NASAThermo,
    ThermoDomainError,
    analytic_h2_oracle,
    h2_he_mixture,
    monatomic_helium,
)
from .opacity import (
    AnalyticGreyOpacity,
    ConstantGreyOpacity,
    PrescribedBandOpacity,
)
from .radiation import (
    DEFAULT_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    LowerFlux,
    LowerTemperature,
    RadiationResult,
    SolveRoute,
    TopIrradiation,
    radiation_core,
    solve_radiation,
)
from .trace import IntegrationTrace, TraceLevel, make_trace

__all__ = [
    "AnalyticGreyOpacity",
    "AnalyticIdealGasThermo",
    "ClosureResult",
    "ConstantGreyOpacity",
    "ConstantGravity",
    "ConstantH2Thermo",
    "ConvergenceMetrics",
    "EnthalpyInversionError",
    "HydrostaticDomainError",
    "IdealH2",
    "IntegrationResult",
    "IntegrationTrace",
    "InverseSquareGravity",
    "MixtureThermo",
    "NASAThermo",
    "PhysicsConfig",
    "PressureGrid",
    "SolverConfig",
    "SolverFailure",
    "TerminalStatus",
    "ThermoDomainError",
    "TraceLevel",
    "analytic_h2_oracle",
    "build_grid",
    "column_enthalpy_per_area",
    "enthalpy_normalized_adiabat",
    "fixed_step",
    "h2_he_mixture",
    "LowerFlux",
    "LowerTemperature",
    "log_pressure_edges",
    "make_trace",
    "mixing_region_labels",
    "mixing_length_flux",
    "monatomic_helium",
    "numerical_isentrope",
    "PrescribedBandOpacity",
    "piecewise_enthalpy_reference",
    "radiation_core",
    "RadiationResult",
    "reconstruct_hydrostatic",
    "reference_enthalpy_residuals",
    "solve_adaptive",
    "solve_radiation",
    "SolveRoute",
    "STEFAN_BOLTZMANN",
    "solve_adaptive_enthalpy",
    "TopIrradiation",
]

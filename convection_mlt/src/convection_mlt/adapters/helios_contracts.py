"""Locked HELIOS radiation-only comparison contracts (Stage 4 point 38)."""

from __future__ import annotations

import math

PINNED_HELIOS_COMMIT = "b0800f9ea4366263241c13bb926e8ca68f266cc5"

# Matched column (radiation-only active variables)
GRAVITY_SI = 15.0  # m s^-2
GRAVITY_CGS = 1500.0  # cm s^-2
F_INT = 300.0  # W m^-2
F_IRR = 120.0  # W m^-2
STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4
T_INT = (F_INT / STEFAN_BOLTZMANN) ** 0.25  # 269.6977849204774 K

# Stage-3 reference closure
STAGE3_DIFFUSIVITY = 1.66
# HELIOS param.dat default; parity reference uses HELIOS-equivalent value when they differ
HELIOS_DEFAULT_DIFFUSIVITY = 2.0

# Unit conversions (documented separately — do not collapse)
FLUX_SI_TO_CGS = 1.0e3  # W m^-2 -> erg s^-1 cm^-2
FLUX_CGS_TO_SI = 1.0e-3
OPACITY_SI_TO_CGS = 10.0  # m^2 kg^-1 -> cm^2 g^-1
OPACITY_CGS_TO_SI = 0.1
GRAVITY_SI_TO_CGS = 100.0
PA_TO_MICROBAR = 10.0  # 1 Pa = 10 microbar (10^-6 bar)
MICROBAR_TO_PA = 0.1
BAR_TO_MICROBAR = 1.0e6  # 1 bar = 1e6 dyne cm^-2
MICROBAR_TO_BAR = 1.0e-6

# Opacity table schema. v3 writes HDF5 pressures in HELIOS CGS (microbar) so
# opac_interpol can compare them to p_lay. v2 had HELIOS kpoints order but bar.
OPACITY_TABLE_SCHEMA_VERSION = "stage4_analytic_grey_v3"
HDF5_PRESSURE_UNIT = "microbar"  # dyne cm^-2 = 1e-6 bar; matches p_lay
TP_PRESSURE_UNIT = "microbar"  # HELIOS tp.dat layer pressure column

# HELIOS host linear indices (kernels.cu / read.py). Fastest axis first.
KPOINTS_LOGICAL_AXES = ["y", "wavelength", "pressure", "temperature"]
KPOINTS_LINEAR_INDEX_ORDER = "y_fastest"
KPOINTS_LINEAR_INDEX_FORMULA = "y + ny*x + ny*nx*p + ny*nx*npress*t"
MEANMOLMASS_LOGICAL_AXES = ["pressure", "temperature"]
MEANMOLMASS_LINEAR_INDEX_FORMULA = "p + npress*t"
RAYLEIGH_LOGICAL_AXES = ["wavelength", "pressure", "temperature"]
RAYLEIGH_LINEAR_INDEX_FORMULA = "x + nx*p + nx*npress*t"


def opacity_layout_metadata() -> dict:
    """HDF5 layout contract matching HELIOS's host index formulas."""
    return {
        "kpoints_logical_axes": list(KPOINTS_LOGICAL_AXES),
        "linear_index_order": KPOINTS_LINEAR_INDEX_ORDER,
        "linear_index_formula": KPOINTS_LINEAR_INDEX_FORMULA,
        "meanmolmass_logical_axes": list(MEANMOLMASS_LOGICAL_AXES),
        "meanmolmass_linear_index_formula": MEANMOLMASS_LINEAR_INDEX_FORMULA,
        "rayleigh_logical_axes": list(RAYLEIGH_LOGICAL_AXES),
        "rayleigh_linear_index_formula": RAYLEIGH_LINEAR_INDEX_FORMULA,
        "hdf5_pressure_unit": HDF5_PRESSURE_UNIT,
        "hdf5_pressure_note": "dyne cm^-2 = microbar; must match HELIOS p_lay",
    }


WRITE_PY_PATCH_NAME = "helios_write_integrated_flux_b0800f9.patch"

HELIOS_PARITY_HEADLINE_MEANS = "coupled_helios_rce_parity"


def _radiation_only_parity(n96: str, n192: str) -> str:
    if n96 == "PASS" and n192 == "PASS":
        return "PASS"
    if n96 == "FAIL" or n192 == "FAIL":
        return "FAIL"
    return "NOT_RUN"


def helios_track_status(
    *,
    adapter_contract: str = "PASS",
    n96: str = "NOT_RUN",
    n192: str = "NOT_RUN",
    coupled_n96: str = "NOT_RUN",
    coupled_n192: str = "NOT_RUN",
) -> dict:
    """HELIOS-track labels.

    ``helios_parity_headline`` is coupled RCE parity only. Radiation-only N=96
    and N=192 agreement is ``helios_radiation_only_parity_status``.
    Coupled N=96 is a pilot; the headline requires N=192. This helper never
    sets ``full_stage4_claim``; the audit builder combines headline with the
    internal numerical track.

    N=96 FAIL with N=192 still NOT_RUN is ``PILOT_FAILED`` (physical pilot
    failure, not an execution/infrastructure failure).
    """
    if coupled_n192 == "PASS" and coupled_n96 == "PASS":
        coupled_overall = "PASS"
        headline = True
    elif coupled_n96 == "FAIL" and coupled_n192 == "NOT_RUN":
        coupled_overall = "PILOT_FAILED"
        headline = False
    elif coupled_n96 == "FAIL" or coupled_n192 == "FAIL":
        coupled_overall = "FAIL"
        headline = False
    elif coupled_n96 == "PASS":
        coupled_overall = "PILOT_ONLY"
        headline = False
    else:
        coupled_overall = "NOT_RUN"
        headline = False
    return {
        "helios_radiation_only_parity_status": _radiation_only_parity(n96, n192),
        "helios_coupled_rce_n96_status": coupled_n96,
        "helios_coupled_rce_n192_status": coupled_n192,
        "helios_coupled_rce_status": coupled_overall,
        "helios_parity_headline": headline,
        "helios_parity_headline_means": HELIOS_PARITY_HEADLINE_MEANS,
        "full_stage4_claim": False,
        "coupled_helios_rce_claimed": headline,
        "helios_adapter_contract_status": adapter_contract,
        "helios_radiation_only_n96_status": n96,
        "helios_radiation_only_n192_status": n192,
    }

PROVENANCE_ONLY = {
    "eos": "ConstantH2Thermo",
    "nabla_ad": 2.0 / 7.0,
}

# HELIOS gas-planet geometry: deepest automatic layer centre must reach 10 bar (1e7 microbar).
HELIOS_GAS_MIN_LAYER_PRESSURE_MICROBAR = 1.0e7


def helios_gas_boa_microbar(
    toa_microbar: float,
    n_layers: int,
    *,
    target_layer_pressure: float = HELIOS_GAS_MIN_LAYER_PRESSURE_MICROBAR,
) -> float:
    """Minimum BOA [microbar] so HELIOS automatic p_lay[0] >= target_layer_pressure."""
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2")
    exponent = 1.0 / (2 * n_layers - 1)
    lo = target_layer_pressure
    hi = max(target_layer_pressure * 2.0, toa_microbar * 1.0e6)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p_bottom_layer = mid * (toa_microbar / mid) ** exponent
        if p_bottom_layer >= target_layer_pressure:
            hi = mid
        else:
            lo = mid
    return hi

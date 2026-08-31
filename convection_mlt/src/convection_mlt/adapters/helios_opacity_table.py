"""Synthetic HELIOS premixed grey opacity table for analytic κ(P)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from convection_mlt.adapters.helios_contracts import (
    BAR_TO_MICROBAR,
    HDF5_PRESSURE_UNIT,
    MICROBAR_TO_BAR,
    OPACITY_TABLE_SCHEMA_VERSION,
    OPACITY_SI_TO_CGS,
    PINNED_HELIOS_COMMIT,
    opacity_layout_metadata,
)
from convection_mlt.opacity import AnalyticGreyOpacity

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


@dataclass(frozen=True)
class HeliosOpacityTable:
    temperatures_k: NDArray[np.float64]
    pressures_bar: NDArray[np.float64]
    wavelengths_cm: NDArray[np.float64]
    gauss_y: NDArray[np.float64]
    kpoints_cgs: NDArray[np.float64]  # (ny, nwave, npress, ntemp)
    rayleigh_cross: NDArray[np.float64]  # (nwave, npress, ntemp)
    mean_mol_mass_kg: NDArray[np.float64]  # (npress, ntemp); HELIOS stores AMU here
    schema_version: str
    metadata: dict

    @property
    def ntemp(self) -> int:
        return int(self.temperatures_k.size)

    @property
    def npress(self) -> int:
        return int(self.pressures_bar.size)


def _require_h5py():
    if h5py is None:
        raise ImportError("h5py is required for HELIOS opacity table I/O")


def analytic_kappa_cgs(
    opacity: AnalyticGreyOpacity,
    temperature: float | NDArray[np.float64],
    pressure_bar: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate κ in cm² g⁻¹ at HELIOS table (T, P_bar) nodes."""
    t = np.atleast_1d(np.asarray(temperature, dtype=np.float64))
    p_pa = np.atleast_1d(np.asarray(pressure_bar, dtype=np.float64)) * 1.0e5
    k_si = opacity.evaluate(t, p_pa)[0]
    return np.atleast_1d(k_si) * OPACITY_SI_TO_CGS


def build_table_arrays(
    opacity: AnalyticGreyOpacity,
    *,
    t_min: float,
    t_max: float,
    p_min_bar: float,
    p_max_bar: float,
    n_temp: int = 32,
    n_press: int = 32,
    mean_mol_mass_amu: float = 2.0,
    n_wave: int = 64,
    w_max_cm: float = 1.0e-1,
) -> HeliosOpacityTable:
    """Build grey premixed table arrays in HELIOS premixed layout."""
    temps = np.geomspace(max(t_min, 50.0), t_max, n_temp)
    press = np.geomspace(max(p_min_bar, 1e-12), p_max_bar, n_press)
    wave = np.geomspace(1.0e-6, w_max_cm, max(n_wave, 2))
    gauss_y = np.array([0.0], dtype=np.float64)
    ny, nw, np_, nt = len(gauss_y), len(wave), len(press), len(temps)
    kpoints = np.zeros((ny, nw, np_, nt), dtype=np.float64)
    for it, t in enumerate(temps):
        for ip, p in enumerate(press):
            k_val = float(analytic_kappa_cgs(opacity, t, p)[0])
            for iw in range(nw):
                kpoints[0, iw, ip, it] = k_val
    rayleigh = np.zeros((nw, np_, nt), dtype=np.float64)
    mean_mass = np.full((np_, nt), mean_mol_mass_amu, dtype=np.float64)
    meta = {
        **opacity_layout_metadata(),
        "schema_version": OPACITY_TABLE_SCHEMA_VERSION,
        "helios_commit": PINNED_HELIOS_COMMIT,
        "opacity_model": "AnalyticGreyOpacity",
        "kappa0_si": opacity.kappa0,
        "P0_Pa": opacity.P0,
        "T0_K": opacity.T0,
        "a": opacity.a,
        "b": opacity.b,
        "pressure_unit": HDF5_PRESSURE_UNIT,
        "opacity_unit": "cm^2 g^-1",
        "temperature_unit": "K",
    }
    return HeliosOpacityTable(
        temperatures_k=temps,
        pressures_bar=press,
        wavelengths_cm=wave,
        gauss_y=gauss_y,
        kpoints_cgs=kpoints,
        rayleigh_cross=rayleigh,
        mean_mol_mass_kg=mean_mass,
        schema_version=OPACITY_TABLE_SCHEMA_VERSION,
        metadata=meta,
    )


def planck_b_lambda_w_m2(temperature_k: float | NDArray[np.float64], wavelength_cm: float) -> float | NDArray[np.float64]:
    """Planck spectral radiance B_lambda [W m^-2 sr^-1 m^-1], lambda in cm."""
    h = 6.62607015e-34
    c = 2.99792458e8
    k_b = 1.380649e-23
    t = np.asarray(temperature_k, dtype=np.float64)
    lam_m = float(wavelength_cm) * 1.0e-2
    x = h * c / (lam_m * k_b * np.maximum(t, 1.0))
    with np.errstate(over="ignore", invalid="ignore"):
        b = (2.0 * h * c**2 / lam_m**5) / (np.expm1(x))
    return b if np.ndim(temperature_k) else float(b)


def bolometric_from_table_bands(
    temperature_k: float | NDArray[np.float64],
    wavelengths_cm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate pi*B_lambda over wavelength bins (2pi hemispheric flux density)."""
    wave = np.asarray(wavelengths_cm, dtype=np.float64)
    inter, deltaw = _wavelength_interfaces(wave)
    t = np.asarray(temperature_k, dtype=np.float64)
    flux = np.zeros_like(t, dtype=np.float64)
    for i in range(len(deltaw)):
        lam = 0.5 * (inter[i] + inter[i + 1])
        flux += np.pi * planck_b_lambda_w_m2(t, lam) * deltaw[i] * 1.0e-2
    return flux


def table_checksum(table: HeliosOpacityTable) -> str:
    h = hashlib.sha256(json.dumps(table.metadata, sort_keys=True).encode())
    h.update(table.kpoints_cgs.tobytes())
    return h.hexdigest()


def _wavelength_interfaces(wave: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    w = np.asarray(wave, dtype=np.float64)
    if w.size == 1:
        w0 = float(w[0])
        inter = np.array([0.5 * w0, 1.5 * w0], dtype=np.float64)
    else:
        inter = np.empty(w.size + 1, dtype=np.float64)
        inter[0] = w[0] - (w[1] - w[0]) / 2.0
        for i in range(w.size - 1):
            inter[i + 1] = 0.5 * (w[i + 1] + w[i])
        inter[-1] = w[-1] + (w[-1] - w[-2]) / 2.0
    deltaw = np.diff(inter)
    return inter, deltaw


def helios_kpoints_flat_index(y: int, x: int, p: int, t: int, *, ny: int, nx: int, npress: int) -> int:
    """HELIOS host index: y + ny*x + ny*nx*p + ny*nx*npress*t (y fastest, T slowest)."""
    return int(y + ny * x + ny * nx * p + ny * nx * npress * t)


def helios_meanmolmass_flat_index(p: int, t: int, *, npress: int) -> int:
    """HELIOS host index: p + npress*t (pressure fastest, T slowest)."""
    return int(p + npress * t)


def helios_rayleigh_flat_index(x: int, p: int, t: int, *, nx: int, npress: int) -> int:
    """HELIOS host index: x + nx*p + nx*npress*t (wavelength fastest, T slowest)."""
    return int(x + nx * p + nx * npress * t)


def unique_kpoints_encoding(y: int, x: int, p: int, t: int) -> float:
    """Axis-unique kpoint so any wrong stride is immediately identifiable."""
    return float(y + 10 * x + 1000 * p + 100000 * t)


def flatten_kpoints_helios_order(kpoints: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten (ny, nwave, npress, ntemp) into HELIOS kpoints order.

    HELIOS reads ``opac_k[y + ny*x + ny*nx*p + ny*nx*npress*t]``. That is
    Fortran-order flattening of the logical ``(y, x, p, t)`` array, equivalently
    C-order of ``kpoints.transpose(3, 2, 1, 0)``.
    """
    k = np.asarray(kpoints, dtype=np.float64)
    if k.ndim != 4:
        raise ValueError("kpoints must have shape (ny, nwave, npress, ntemp)")
    return np.asarray(k, dtype=np.float64).ravel(order="F")


def unflatten_kpoints_helios_order(
    flat: NDArray[np.float64], shape: tuple[int, int, int, int]
) -> NDArray[np.float64]:
    """Inverse of ``flatten_kpoints_helios_order``."""
    return np.asarray(flat, dtype=np.float64).reshape(shape, order="F")


def flatten_kpoints_numpy_c_order(kpoints: NDArray[np.float64]) -> NDArray[np.float64]:
    """Legacy C-order flatten of (ny, nwave, npress, ntemp). Not HELIOS native."""
    return np.asarray(kpoints, dtype=np.float64).ravel()


def flatten_meanmolmass_helios_order(mean_mass: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten (npress, ntemp) with p fastest: ``p + npress*t``."""
    a = np.asarray(mean_mass, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("meanmolmass must have shape (npress, ntemp)")
    return a.ravel(order="F")


def unflatten_meanmolmass_helios_order(
    flat: NDArray[np.float64], npress: int, ntemp: int
) -> NDArray[np.float64]:
    return np.asarray(flat, dtype=np.float64).reshape((npress, ntemp), order="F")


def flatten_rayleigh_helios_order(rayleigh: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten (nwave, npress, ntemp) with wavelength fastest: ``x + nx*p + nx*npress*t``."""
    a = np.asarray(rayleigh, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("Rayleigh cross-sections must have shape (nwave, npress, ntemp)")
    return a.ravel(order="F")


def unflatten_rayleigh_helios_order(
    flat: NDArray[np.float64], nwave: int, npress: int, ntemp: int
) -> NDArray[np.float64]:
    return np.asarray(flat, dtype=np.float64).reshape((nwave, npress, ntemp), order="F")


def helios_logp_table_index(
    p_atm_microbar: float,
    kpress_microbar: NDArray[np.float64],
) -> float:
    """Fractional pressure index used by HELIOS ``opac_interpol`` (clamped)."""
    k = np.asarray(kpress_microbar, dtype=np.float64)
    npress = int(k.size)
    if npress < 2:
        raise ValueError("kpress must have at least 2 nodes")
    delta = (np.log10(k[-1]) - np.log10(k[0])) / (npress - 1.0)
    p = (np.log10(float(p_atm_microbar)) - np.log10(k[0])) / delta
    return float(np.clip(p, 0.001, npress - 1.001))


def _empty_table_shell(
    *,
    temps: NDArray[np.float64],
    press: NDArray[np.float64],
    wave: NDArray[np.float64],
    mean_mol_mass_amu: float,
    metadata: dict,
) -> HeliosOpacityTable:
    gauss_y = np.array([0.0], dtype=np.float64)
    ny, nw, np_, nt = 1, len(wave), len(press), len(temps)
    return HeliosOpacityTable(
        temperatures_k=temps,
        pressures_bar=press,
        wavelengths_cm=wave,
        gauss_y=gauss_y,
        kpoints_cgs=np.zeros((ny, nw, np_, nt), dtype=np.float64),
        rayleigh_cross=np.zeros((nw, np_, nt), dtype=np.float64),
        mean_mol_mass_kg=np.full((np_, nt), mean_mol_mass_amu, dtype=np.float64),
        schema_version=OPACITY_TABLE_SCHEMA_VERSION,
        metadata=metadata,
    )


def build_constant_opacity_table(
    kappa_si: float,
    *,
    t_min: float,
    t_max: float,
    p_min_bar: float,
    p_max_bar: float,
    n_temp: int = 32,
    n_press: int = 32,
    mean_mol_mass_amu: float = 2.0,
    n_wave: int = 64,
    w_max_cm: float = 1.0e-1,
) -> HeliosOpacityTable:
    """Grey table with identical κ at every (T, P, λ) node."""
    temps = np.geomspace(max(t_min, 50.0), t_max, n_temp)
    press = np.geomspace(max(p_min_bar, 1e-12), p_max_bar, n_press)
    wave = np.geomspace(1.0e-6, w_max_cm, max(n_wave, 2))
    k_cgs = float(kappa_si) * OPACITY_SI_TO_CGS
    meta = {
        **opacity_layout_metadata(),
        "schema_version": OPACITY_TABLE_SCHEMA_VERSION,
        "helios_commit": PINNED_HELIOS_COMMIT,
        "opacity_model": "ConstantGreyOpacity",
        "kappa_si": float(kappa_si),
        "kappa_cgs": k_cgs,
        "pressure_unit": HDF5_PRESSURE_UNIT,
        "opacity_unit": "cm^2 g^-1",
        "temperature_unit": "K",
        "diagnostic": "constant_opacity_control",
    }
    table = _empty_table_shell(
        temps=temps, press=press, wave=wave,
        mean_mol_mass_amu=mean_mol_mass_amu, metadata=meta,
    )
    table.kpoints_cgs[...] = k_cgs
    return table


def build_pressure_tagged_table(
    *,
    t_min: float,
    t_max: float,
    p_min_bar: float,
    p_max_bar: float,
    n_temp: int = 32,
    n_press: int = 32,
    mean_mol_mass_amu: float = 2.0,
    n_wave: int = 64,
    w_max_cm: float = 1.0e-1,
    kappa_scale_cgs: float = 1.0,
) -> HeliosOpacityTable:
    """Unmistakable P-only table: κ_cgs = kappa_scale_cgs * P_bar, independent of T, λ."""
    temps = np.geomspace(max(t_min, 50.0), t_max, n_temp)
    press = np.geomspace(max(p_min_bar, 1e-12), p_max_bar, n_press)
    wave = np.geomspace(1.0e-6, w_max_cm, max(n_wave, 2))
    meta = {
        **opacity_layout_metadata(),
        "schema_version": OPACITY_TABLE_SCHEMA_VERSION,
        "helios_commit": PINNED_HELIOS_COMMIT,
        "opacity_model": "PressureTagged",
        "kappa_cgs": "kappa_scale_cgs * P_bar",
        "kappa_scale_cgs": float(kappa_scale_cgs),
        "pressure_unit": HDF5_PRESSURE_UNIT,
        "opacity_unit": "cm^2 g^-1",
        "temperature_unit": "K",
        "diagnostic": "pressure_tagged_axis_probe",
    }
    table = _empty_table_shell(
        temps=temps, press=press, wave=wave,
        mean_mol_mass_amu=mean_mol_mass_amu, metadata=meta,
    )
    for ip, p in enumerate(press):
        table.kpoints_cgs[:, :, ip, :] = float(kappa_scale_cgs) * float(p)
    return table


def build_unique_axis_table(
    *,
    n_y: int = 3,
    n_wave: int = 4,
    n_press: int = 5,
    n_temp: int = 2,
) -> HeliosOpacityTable:
    """Table with unique values on every (y, x, p, t) and (p, t) / (x, p, t) axis."""
    temps = np.linspace(200.0, 800.0, n_temp)
    press = np.geomspace(1.0e-3, 10.0, n_press)
    wave = np.geomspace(1.0e-5, 1.0e-2, n_wave)
    gauss_y = np.linspace(0.0, 1.0, n_y)
    kpoints = np.empty((n_y, n_wave, n_press, n_temp), dtype=np.float64)
    rayleigh = np.empty((n_wave, n_press, n_temp), dtype=np.float64)
    mean_mass = np.empty((n_press, n_temp), dtype=np.float64)
    for t in range(n_temp):
        for p in range(n_press):
            mean_mass[p, t] = 1.0 + p + 1000.0 * t
            for x in range(n_wave):
                rayleigh[x, p, t] = x + 100.0 * p + 10000.0 * t
                for y in range(n_y):
                    kpoints[y, x, p, t] = unique_kpoints_encoding(y, x, p, t)
    meta = {
        **opacity_layout_metadata(),
        "schema_version": OPACITY_TABLE_SCHEMA_VERSION,
        "helios_commit": PINNED_HELIOS_COMMIT,
        "opacity_model": "UniqueAxisEncoding",
        "kpoints_encoding": "y + 10*x + 1000*p + 100000*t",
        "meanmolmass_encoding": "1 + p + 1000*t",
        "rayleigh_encoding": "x + 100*p + 10000*t",
        "pressure_unit": HDF5_PRESSURE_UNIT,
        "opacity_unit": "cm^2 g^-1",
        "temperature_unit": "K",
        "diagnostic": "hdf5_linear_index_roundtrip",
    }
    return HeliosOpacityTable(
        temperatures_k=temps,
        pressures_bar=press,
        wavelengths_cm=wave,
        gauss_y=gauss_y,
        kpoints_cgs=kpoints,
        rayleigh_cross=rayleigh,
        mean_mol_mass_kg=mean_mass,
        schema_version=OPACITY_TABLE_SCHEMA_VERSION,
        metadata=meta,
    )


def write_helios_opacity_hdf5(
    path: str | Path,
    table: HeliosOpacityTable,
    *,
    flatten: str = "helios",
) -> str:
    """Write HELIOS-compatible premixed HDF5; return checksum.

    Default ``flatten='helios'`` is Fortran-order of ``(y, x, p, t)``, matching
    HELIOS ``i = y + ny*x + ny*nx*p + ny*nx*npress*t``. ``flatten='numpy_c'`` is
    retained only to demonstrate the legacy bug; do not use it for live tables.
    meanmolmass and Rayleigh always use their HELIOS host formulas.
    """
    _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nw, np_, nt = table.kpoints_cgs.shape
    if flatten == "helios":
        flat = flatten_kpoints_helios_order(table.kpoints_cgs)
    elif flatten == "numpy_c":
        flat = flatten_kpoints_numpy_c_order(table.kpoints_cgs)
    else:
        raise ValueError(f"unknown flatten mode: {flatten}")
    if flat.size != ny * nw * np_ * nt:
        raise ValueError(f"kpoints size {flat.size} != {ny * nw * np_ * nt}")
    mean_flat = flatten_meanmolmass_helios_order(table.mean_mol_mass_kg)
    ray_flat = flatten_rayleigh_helios_order(table.rayleigh_cross)
    if mean_flat.size != np_ * nt:
        raise ValueError(f"meanmolmass size {mean_flat.size} != {np_ * nt}")
    if ray_flat.size != nw * np_ * nt:
        raise ValueError(f"Rayleigh size {ray_flat.size} != {nw * np_ * nt}")
    inter_wave, deltaw = _wavelength_interfaces(table.wavelengths_cm)
    layout = opacity_layout_metadata()
    meta = {**table.metadata, **layout, "kpoints_flatten": flatten}
    with h5py.File(path, "w") as f:
        f.create_dataset("kpoints", data=flat)
        f.create_dataset("weighted Rayleigh cross-sections", data=ray_flat)
        f.create_dataset("meanmolmass", data=mean_flat)
        f.create_dataset("center wavelengths", data=table.wavelengths_cm)
        f.create_dataset("interface wavelengths", data=inter_wave)
        f.create_dataset("wavelength width of bins", data=deltaw)
        f.create_dataset("ypoints", data=table.gauss_y)
        f.create_dataset("temperatures", data=table.temperatures_k)
        f.create_dataset("pressures", data=np.asarray(table.pressures_bar, dtype=np.float64) * BAR_TO_MICROBAR)
        f.attrs["schema_version"] = table.schema_version
        f.attrs["hdf5_pressure_unit"] = HDF5_PRESSURE_UNIT
        f.attrs["kpoints_flatten"] = flatten
        f.attrs["linear_index_order"] = layout["linear_index_order"]
        f.attrs["linear_index_formula"] = layout["linear_index_formula"]
        f.attrs["kpoints_logical_axes"] = json.dumps(layout["kpoints_logical_axes"])
        f.attrs["meanmolmass_linear_index_formula"] = layout["meanmolmass_linear_index_formula"]
        f.attrs["rayleigh_linear_index_formula"] = layout["rayleigh_linear_index_formula"]
        f.attrs["metadata_json"] = json.dumps(meta)
    return table_checksum(table)


def read_helios_opacity_hdf5(path: str | Path) -> HeliosOpacityTable:
    _require_h5py()
    with h5py.File(path, "r") as f:
        temps = np.asarray(f["temperatures"][:], dtype=np.float64)
        press = np.asarray(f["pressures"][:], dtype=np.float64)
        wave = np.asarray(f["center wavelengths"][:], dtype=np.float64)
        gauss_y = np.asarray(f["ypoints"][:], dtype=np.float64)
        flat = np.asarray(f["kpoints"][:], dtype=np.float64)
        ray_flat = np.asarray(f["weighted Rayleigh cross-sections"][:], dtype=np.float64)
        mean_flat = np.asarray(f["meanmolmass"][:], dtype=np.float64)
        meta = json.loads(f.attrs.get("metadata_json", "{}"))
        flatten = str(f.attrs.get("kpoints_flatten", meta.get("kpoints_flatten", "helios")))
        pressure_unit = str(
            f.attrs.get("hdf5_pressure_unit", meta.get("hdf5_pressure_unit", meta.get("pressure_unit", "bar")))
        )
    if pressure_unit in ("microbar", "dyne cm^-2", "dyne/cm^2"):
        press = press * MICROBAR_TO_BAR
    elif pressure_unit != "bar":
        raise ValueError(f"unknown HDF5 pressure unit: {pressure_unit}")
    ny, nw = len(gauss_y), len(wave)
    np_, nt = len(press), len(temps)
    if flat.size != ny * nw * np_ * nt:
        raise ValueError(f"kpoints size {flat.size} != {ny * nw * np_ * nt}")
    if flatten == "numpy_c":
        kpoints = np.asarray(flat, dtype=np.float64).reshape((ny, nw, np_, nt), order="C")
    else:
        kpoints = unflatten_kpoints_helios_order(flat, (ny, nw, np_, nt))
    mean_mass = unflatten_meanmolmass_helios_order(mean_flat, np_, nt)
    rayleigh = unflatten_rayleigh_helios_order(ray_flat, nw, np_, nt)
    return HeliosOpacityTable(
        temperatures_k=temps,
        pressures_bar=press,
        wavelengths_cm=wave,
        gauss_y=gauss_y,
        kpoints_cgs=kpoints,
        rayleigh_cross=rayleigh,
        mean_mol_mass_kg=mean_mass,
        schema_version=str(meta.get("schema_version", "")),
        metadata=meta,
    )


def _locate_axis(x: NDArray[np.float64], value: float) -> tuple[int, int, float]:
    if value <= x[0]:
        return 0, 0, 0.0
    if value >= x[-1]:
        return len(x) - 1, len(x) - 1, 0.0
    j = int(np.searchsorted(x, value) - 1)
    j = max(0, min(j, len(x) - 2))
    x0, x1 = float(x[j]), float(x[j + 1])
    w = 0.0 if x1 == x0 else (value - x0) / (x1 - x0)
    return j, j + 1, w


def interpolate_opacity_cgs(
    table: HeliosOpacityTable,
    temperature_k: float,
    pressure_bar: float,
    *,
    allow_extrapolation: bool = False,
) -> float:
    """Bilinear interpolation in log-T and log-P (HELIOS-style table lookup)."""
    t = float(temperature_k)
    p = float(pressure_bar)
    if not allow_extrapolation:
        if t < table.temperatures_k[0] or t > table.temperatures_k[-1]:
            raise ValueError(f"temperature {t} outside table hull")
        if p < table.pressures_bar[0] or p > table.pressures_bar[-1]:
            raise ValueError(f"pressure {p} bar outside table hull")
    logt = np.log(np.maximum(table.temperatures_k, 1.0))
    logp = np.log(np.maximum(table.pressures_bar, 1e-300))
    lt = np.log(max(t, 1.0))
    lp = np.log(max(p, 1e-300))
    i0, i1, wt = _locate_axis(logt, lt)
    j0, j1, wp = _locate_axis(logp, lp)
    k = table.kpoints_cgs[0, 0]
    v00 = k[j0, i0]
    v01 = k[j0, i1]
    v10 = k[j1, i0]
    v11 = k[j1, i1]
    v0 = (1.0 - wt) * v00 + wt * v01
    v1 = (1.0 - wt) * v10 + wt * v11
    return float((1.0 - wp) * v0 + wp * v1)


def interpolate_opacity_si(
    table: HeliosOpacityTable,
    temperature_k: float,
    pressure_pa: float,
    **kwargs,
) -> float:
    from convection_mlt.adapters.helios_contracts import OPACITY_CGS_TO_SI

    p_bar = pressure_pa / 1.0e5
    return interpolate_opacity_cgs(table, temperature_k, p_bar, **kwargs) * OPACITY_CGS_TO_SI

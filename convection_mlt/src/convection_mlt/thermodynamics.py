"""Thermodynamic providers for constant-cp H2, analytic oracles, NASA/CEA, and mixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validate import positive, temperatures

R_UNIVERSAL = 8.31446261815324  # J mol^-1 K^-1
H2_MOLAR_MASS = 2.01588e-3  # kg mol^-1
HE_MOLAR_MASS = 4.002602e-3  # kg mol^-1
T_REF = 298.15  # K
P_REF = 1.0e5  # Pa


class ThermoDomainError(ValueError):
    """Temperature (or other state) outside the provider's valid domain."""


class EnthalpyInversionError(ValueError):
    """Enthalpy could not be inverted to a unique finite temperature."""


@runtime_checkable
class ThermoProvider(Protocol):
    molar_mass: float
    gas_constant: float
    t_ref: float
    p_ref: float

    def specific_heat(self, temperature: ArrayLike) -> NDArray[np.float64]: ...
    def enthalpy(self, temperature: ArrayLike) -> NDArray[np.float64]: ...
    def psi(self, temperature: ArrayLike) -> NDArray[np.float64]: ...
    def entropy(self, temperature: ArrayLike, pressure: ArrayLike) -> NDArray[np.float64]: ...
    def nabla_ad_at(self, temperature: ArrayLike) -> NDArray[np.float64]: ...
    def density(self, pressure: ArrayLike, temperature: ArrayLike) -> NDArray[np.float64]: ...
    def invert_enthalpy(self, enthalpy: ArrayLike) -> NDArray[np.float64]: ...
    def potential_temperature(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> NDArray[np.float64]: ...
    def as_metadata(self) -> dict[str, Any]: ...


def _as_float_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _match_shapes(a: NDArray[np.float64], b: NDArray[np.float64], name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name} shapes must match: {a.shape} vs {b.shape}")


def invert_monotonic(
    function,
    targets: ArrayLike,
    t_min: float,
    t_max: float,
    *,
    name: str = "value",
) -> NDArray[np.float64]:
    """Invert a strictly increasing scalar function on [t_min, t_max]."""
    target = _as_float_array(name, targets)
    flat = target.ravel()
    out = np.empty_like(flat)
    f_lo = float(function(np.asarray([t_min]))[0])
    f_hi = float(function(np.asarray([t_max]))[0])
    if not (f_hi > f_lo):
        raise EnthalpyInversionError(f"{name} is not strictly increasing on domain")
    for index, value in enumerate(flat):
        if not np.isfinite(value):
            raise EnthalpyInversionError(f"{name} target is nonfinite")
        if value < f_lo - 1.0e-12 * max(1.0, abs(f_lo)) or value > f_hi + 1.0e-12 * max(
            1.0, abs(f_hi)
        ):
            raise EnthalpyInversionError(
                f"{name}={value} outside invertible range [{f_lo}, {f_hi}]"
            )
        lo = t_min
        hi = t_max
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = float(function(np.asarray([mid]))[0])
            if f_mid < value:
                lo = mid
            else:
                hi = mid
        out[index] = 0.5 * (lo + hi)
    return out.reshape(target.shape)


def _density_ideal(
    gas_constant: float, pressure: ArrayLike, temperature: ArrayLike
) -> NDArray[np.float64]:
    p = _as_float_array("pressure", pressure)
    t = temperatures(temperature)
    _match_shapes(p, t, "pressure/temperature")
    if np.any(p <= 0.0):
        raise ValueError("pressure must be positive")
    return p / (gas_constant * t)


@dataclass(frozen=True)
class ConstantH2Thermo:
    """Calorically perfect H2; Stage 0/1 regression provider."""

    molar_mass: float = H2_MOLAR_MASS
    t_ref: float = T_REF
    p_ref: float = P_REF

    def __post_init__(self) -> None:
        positive("molar_mass", self.molar_mass)
        positive("t_ref", self.t_ref)
        positive("p_ref", self.p_ref)

    @property
    def t_min(self) -> float:
        return 1.0e-3

    @property
    def t_max(self) -> float:
        return 1.0e6

    @property
    def gas_constant(self) -> float:
        return R_UNIVERSAL / self.molar_mass

    @property
    def cp(self) -> float:
        """Legacy scalar specific heat for Stage 0/1."""
        return 3.5 * self.gas_constant

    @property
    def nabla_ad(self) -> float:
        """Legacy scalar adiabatic gradient for Stage 0/1."""
        return self.gas_constant / self.cp

    def specific_heat(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        return np.full(t.shape, self.cp, dtype=float)

    def enthalpy(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        return self.cp * (t - self.t_ref)

    def psi(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        if np.any(t <= 0.0):
            raise ThermoDomainError("temperature must be positive")
        return self.cp * np.log(t / self.t_ref)

    def entropy(self, temperature: ArrayLike, pressure: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        if np.any(p <= 0.0):
            raise ValueError("pressure must be positive")
        return self.psi(t) - self.gas_constant * np.log(p / self.p_ref)

    def nabla_ad_at(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        return np.full(t.shape, self.nabla_ad, dtype=float)

    def density(self, pressure: ArrayLike, temperature: ArrayLike) -> NDArray[np.float64]:
        return _density_ideal(self.gas_constant, pressure, temperature)

    def invert_enthalpy(self, enthalpy: ArrayLike) -> NDArray[np.float64]:
        h = _as_float_array("enthalpy", enthalpy)
        return self.t_ref + h / self.cp

    def potential_temperature(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        return t * (self.p_ref / p) ** self.nabla_ad

    def scale_height(self, temperature: ArrayLike, gravity: float) -> NDArray[np.float64]:
        positive("gravity", gravity)
        return self.gas_constant * temperatures(temperature) / gravity

    def as_metadata(self) -> dict[str, Any]:
        return {
            "provider": "ConstantH2Thermo",
            "molar_mass": self.molar_mass,
            "gas_constant": self.gas_constant,
            "cp": self.cp,
            "nabla_ad": self.nabla_ad,
            "t_ref": self.t_ref,
            "p_ref": self.p_ref,
        }


# Backward-compatible alias used throughout Stage 0/1.
IdealH2 = ConstantH2Thermo


@dataclass(frozen=True)
class AnalyticIdealGasThermo:
    """Differentiable ideal-gas oracle with optional vibrational excitation (Option A)."""

    molar_mass: float
    degrees_of_freedom_trans_rot: float = 5.0
    vibrational_temperature: float | None = None
    t_min: float = 50.0
    t_max: float = 6000.0
    t_ref: float = T_REF
    p_ref: float = P_REF

    def __post_init__(self) -> None:
        positive("molar_mass", self.molar_mass)
        positive("degrees_of_freedom_trans_rot", self.degrees_of_freedom_trans_rot)
        positive("t_min", self.t_min)
        positive("t_max", self.t_max)
        if self.t_max <= self.t_min:
            raise ValueError("t_max must exceed t_min")
        if self.vibrational_temperature is not None:
            positive("vibrational_temperature", self.vibrational_temperature)

    @property
    def gas_constant(self) -> float:
        return R_UNIVERSAL / self.molar_mass

    def _ensure_domain(self, temperature: NDArray[np.float64]) -> None:
        if np.any(temperature < self.t_min) or np.any(temperature > self.t_max):
            raise ThermoDomainError(
                f"temperature outside AnalyticIdealGas domain "
                f"[{self.t_min}, {self.t_max}] K"
            )

    def specific_heat(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        self._ensure_domain(t)
        cp = np.full(
            t.shape,
            (self.degrees_of_freedom_trans_rot / 2.0 + 1.0) * self.gas_constant,
            dtype=float,
        )
        if self.vibrational_temperature is not None:
            theta = self.vibrational_temperature
            x = theta / t
            ex = np.exp(x)
            cp = cp + self.gas_constant * (x**2) * ex / (ex - 1.0) ** 2
        return cp

    def enthalpy(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        self._ensure_domain(t)
        base = (self.degrees_of_freedom_trans_rot / 2.0 + 1.0) * self.gas_constant
        h = base * (t - self.t_ref)
        if self.vibrational_temperature is not None:
            theta = self.vibrational_temperature
            h = h + self.gas_constant * (
                theta / (np.exp(theta / t) - 1.0)
                - theta / (np.exp(theta / self.t_ref) - 1.0)
            )
        return h

    def psi(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        self._ensure_domain(t)
        base = (self.degrees_of_freedom_trans_rot / 2.0 + 1.0) * self.gas_constant
        psi = base * np.log(t / self.t_ref)
        if self.vibrational_temperature is not None:
            theta = self.vibrational_temperature
            # ∫ (θ/T)^2 e^{θ/T}/(e^{θ/T}-1)^2 (R) dln? Actually ∫ cp_vib/T dT
            # = R [ θ/T /(e^{θ/T}-1) - ln(1-e^{-θ/T}) ] evaluated difference.
            def vib_entropy_over_r(temp: NDArray[np.float64]) -> NDArray[np.float64]:
                x = theta / temp
                return x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x))

            psi = psi + self.gas_constant * (
                vib_entropy_over_r(t) - vib_entropy_over_r(np.asarray(self.t_ref))
            )
        return psi

    def entropy(self, temperature: ArrayLike, pressure: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        if np.any(p <= 0.0):
            raise ValueError("pressure must be positive")
        return self.psi(t) - self.gas_constant * np.log(p / self.p_ref)

    def nabla_ad_at(self, temperature: ArrayLike) -> NDArray[np.float64]:
        return self.gas_constant / self.specific_heat(temperature)

    def density(self, pressure: ArrayLike, temperature: ArrayLike) -> NDArray[np.float64]:
        return _density_ideal(self.gas_constant, pressure, temperature)

    def invert_enthalpy(self, enthalpy: ArrayLike) -> NDArray[np.float64]:
        return invert_enthalpy_newton(self, enthalpy, t_min=self.t_min, t_max=self.t_max)

    def potential_temperature(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        target_psi = self.psi(t) - self.gas_constant * np.log(p / self.p_ref)
        return invert_psi_newton(self, target_psi, t_min=self.t_min, t_max=self.t_max)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "provider": "AnalyticIdealGasThermo",
            "molar_mass": self.molar_mass,
            "gas_constant": self.gas_constant,
            "degrees_of_freedom_trans_rot": self.degrees_of_freedom_trans_rot,
            "vibrational_temperature": self.vibrational_temperature,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "t_ref": self.t_ref,
            "p_ref": self.p_ref,
        }


def analytic_h2_oracle(**kwargs: Any) -> AnalyticIdealGasThermo:
    """H2-like analytic oracle with rotational DOF=5 and optional vibration."""
    defaults = {
        "molar_mass": H2_MOLAR_MASS,
        "degrees_of_freedom_trans_rot": 5.0,
        "vibrational_temperature": 6215.0,  # characteristic vib temp ~H2, oracle only
        "t_min": 200.0,
        "t_max": 6000.0,
    }
    defaults.update(kwargs)
    return AnalyticIdealGasThermo(**defaults)


def monatomic_helium(**kwargs: Any) -> AnalyticIdealGasThermo:
    """Exact monatomic He: cp = (5/2) R over the Stage 2 temperature range."""
    defaults = {
        "molar_mass": HE_MOLAR_MASS,
        "degrees_of_freedom_trans_rot": 3.0,
        "vibrational_temperature": None,
        "t_min": 200.0,
        "t_max": 6000.0,
    }
    defaults.update(kwargs)
    return AnalyticIdealGasThermo(**defaults)


def _nasa7_cp_over_r(a: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        a[..., 0]
        + a[..., 1] * t
        + a[..., 2] * t**2
        + a[..., 3] * t**3
        + a[..., 4] * t**4
    )


def _nasa7_h_over_rt(a: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        a[..., 0]
        + a[..., 1] * t / 2.0
        + a[..., 2] * t**2 / 3.0
        + a[..., 3] * t**3 / 4.0
        + a[..., 4] * t**4 / 5.0
        + a[..., 5] / t
    )


def _nasa7_s_over_r(a: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        a[..., 0] * np.log(t)
        + a[..., 1] * t
        + a[..., 2] * t**2 / 2.0
        + a[..., 3] * t**3 / 3.0
        + a[..., 4] * t**4 / 4.0
        + a[..., 6]
    )



def invert_enthalpy_newton(
    thermo,
    enthalpy: ArrayLike,
    *,
    t_min: float,
    t_max: float,
) -> NDArray[np.float64]:
    """Vectorized Newton inversion of h(T) using cp = dh/dT."""
    h = _as_float_array("enthalpy", enthalpy)
    h_lo = float(thermo.enthalpy(np.asarray([t_min]))[0])
    h_hi = float(thermo.enthalpy(np.asarray([t_max]))[0])
    if np.any(~np.isfinite(h)):
        raise EnthalpyInversionError("enthalpy target is nonfinite")
    if np.any(h < h_lo - 1.0e-12 * max(1.0, abs(h_lo))) or np.any(
        h > h_hi + 1.0e-12 * max(1.0, abs(h_hi))
    ):
        raise EnthalpyInversionError(
            f"enthalpy outside invertible range [{h_lo}, {h_hi}]"
        )
    # Linear seed using mid-domain cp.
    t_mid = 0.5 * (t_min + t_max)
    cp_mid = float(thermo.specific_heat(np.asarray([t_mid]))[0])
    t = np.clip(t_mid + (h - float(thermo.enthalpy(np.asarray([t_mid]))[0])) / cp_mid, t_min, t_max)
    for _ in range(25):
        residual = thermo.enthalpy(t) - h
        if float(np.max(np.abs(residual))) <= 1.0e-14 * max(1.0, float(np.max(np.abs(h)))):
            break
        cp = thermo.specific_heat(t)
        t = np.clip(t - residual / cp, t_min, t_max)
    # Final safeguard with bisection on any lagging points.
    residual = thermo.enthalpy(t) - h
    lagging = np.abs(residual) > 1.0e-12 * np.maximum(1.0, np.abs(h))
    if np.any(lagging):
        t_fix = invert_monotonic(thermo.enthalpy, h[lagging], t_min, t_max, name="enthalpy")
        t = t.copy()
        t[lagging] = t_fix
    return t


def invert_psi_newton(
    thermo,
    target_psi: ArrayLike,
    *,
    t_min: float,
    t_max: float,
) -> NDArray[np.float64]:
    """Vectorized Newton inversion of Ψ(T) using dΨ/dT = cp/T."""
    target = _as_float_array("psi", target_psi)
    psi_lo = float(thermo.psi(np.asarray([t_min]))[0])
    psi_hi = float(thermo.psi(np.asarray([t_max]))[0])
    if np.any(~np.isfinite(target)):
        raise EnthalpyInversionError("psi target is nonfinite")
    if np.any(target < psi_lo - 1.0e-12 * max(1.0, abs(psi_lo))) or np.any(
        target > psi_hi + 1.0e-12 * max(1.0, abs(psi_hi))
    ):
        raise EnthalpyInversionError(
            f"psi outside invertible range [{psi_lo}, {psi_hi}]"
        )
    t_mid = 0.5 * (t_min + t_max)
    cp_mid = float(thermo.specific_heat(np.asarray([t_mid]))[0])
    psi_mid = float(thermo.psi(np.asarray([t_mid]))[0])
    t = np.clip(t_mid * np.exp((target - psi_mid) / cp_mid), t_min, t_max)
    for _ in range(40):
        residual = thermo.psi(t) - target
        if float(np.max(np.abs(residual))) <= 1.0e-16 * max(1.0, float(np.max(np.abs(target)))):
            break
        dpsi = thermo.specific_heat(t) / t
        t = np.clip(t - residual / dpsi, t_min, t_max)
    residual = thermo.psi(t) - target
    lagging = np.abs(residual) > 1.0e-14 * np.maximum(1.0, np.abs(target))
    if np.any(lagging):
        t_fix = invert_monotonic(thermo.psi, target[lagging], t_min, t_max, name="psi")
        t = t.copy()
        t[lagging] = t_fix
    return t

@dataclass(frozen=True)
class NASAThermo:
    """NASA7 polynomial thermodynamics for a single species (production Stage 2 H2)."""

    species: str
    molar_mass: float
    intervals: tuple[tuple[float, float, tuple[float, ...]], ...]
    source: str
    cea_note: str
    t_ref: float = T_REF
    p_ref: float = P_REF
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive("molar_mass", self.molar_mass)
        if not self.intervals:
            raise ValueError("NASAThermo requires at least one temperature interval")
        for t_min, t_max, coeffs in self.intervals:
            if t_max <= t_min:
                raise ValueError("invalid NASA temperature interval")
            if len(coeffs) != 7:
                raise ValueError("NASA7 intervals require 7 coefficients")

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "NASAThermo":
        if path is None:
            data_file = resources.files("convection_mlt.thermo_data").joinpath(
                "h2_nasa7_tpis78.json"
            )
            payload = json.loads(data_file.read_text(encoding="utf-8"))
        else:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        intervals = tuple(
            (
                float(item["t_min_k"]),
                float(item["t_max_k"]),
                tuple(float(c) for c in item["coefficients"]),
            )
            for item in payload["intervals"]
        )
        return cls(
            species=str(payload["species"]),
            molar_mass=float(payload["molar_mass_kg_per_mol"]),
            intervals=intervals,
            source=str(payload["source"]),
            cea_note=str(payload.get("cea_note", "")),
            t_ref=float(payload.get("t_ref_k", T_REF)),
            p_ref=float(payload.get("p_ref_pa", P_REF)),
            provenance={
                "citation": payload.get("citation"),
                "format": payload.get("format"),
                "temperature_ranges_k": payload.get("temperature_ranges_k"),
                "units": payload.get("units"),
            },
        )

    @property
    def gas_constant(self) -> float:
        return R_UNIVERSAL / self.molar_mass

    @property
    def t_min(self) -> float:
        return float(self.intervals[0][0])

    @property
    def t_max(self) -> float:
        return float(self.intervals[-1][1])

    def _coeffs_for(self, temperature: NDArray[np.float64]) -> NDArray[np.float64]:
        if np.any(temperature < self.t_min) or np.any(temperature > self.t_max):
            raise ThermoDomainError(
                f"{self.species} temperature outside NASA domain "
                f"[{self.t_min}, {self.t_max}] K"
            )
        # Interior breakpoints belong to the higher interval; the final
        # interval includes its upper bound.
        coeffs = np.empty(temperature.shape + (7,), dtype=float)
        assigned = np.zeros(temperature.shape, dtype=bool)
        for index, (t_min, t_max, values) in enumerate(self.intervals):
            a = np.asarray(values, dtype=float)
            if index == len(self.intervals) - 1:
                mask = (temperature >= t_min) & (temperature <= t_max)
            else:
                mask = (temperature >= t_min) & (temperature < t_max)
            coeffs[mask] = a
            assigned |= mask
        if not np.all(assigned):
            raise ThermoDomainError(f"{self.species}: failed to assign NASA interval")
        return coeffs

    def specific_heat(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        a = self._coeffs_for(t)
        cp_over_r = _nasa7_cp_over_r(a, t)
        return (R_UNIVERSAL * cp_over_r) / self.molar_mass

    def _h_molar_absolute(self, temperature: NDArray[np.float64]) -> NDArray[np.float64]:
        a = self._coeffs_for(temperature)
        h_over_rt = _nasa7_h_over_rt(a, temperature)
        return R_UNIVERSAL * temperature * h_over_rt

    def enthalpy(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        h_ref = self._h_molar_absolute(np.asarray([self.t_ref], dtype=float))[0]
        h = self._h_molar_absolute(t) - h_ref
        return h / self.molar_mass

    def _s_molar_standard(self, temperature: NDArray[np.float64]) -> NDArray[np.float64]:
        a = self._coeffs_for(temperature)
        s_over_r = _nasa7_s_over_r(a, temperature)
        return R_UNIVERSAL * s_over_r

    def psi(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        s_ref = self._s_molar_standard(np.asarray([self.t_ref], dtype=float))[0]
        return (self._s_molar_standard(t) - s_ref) / self.molar_mass

    def entropy(self, temperature: ArrayLike, pressure: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        if np.any(p <= 0.0):
            raise ValueError("pressure must be positive")
        return self.psi(t) - self.gas_constant * np.log(p / self.p_ref)

    def nabla_ad_at(self, temperature: ArrayLike) -> NDArray[np.float64]:
        return self.gas_constant / self.specific_heat(temperature)

    def density(self, pressure: ArrayLike, temperature: ArrayLike) -> NDArray[np.float64]:
        return _density_ideal(self.gas_constant, pressure, temperature)

    def invert_enthalpy(self, enthalpy: ArrayLike) -> NDArray[np.float64]:
        return invert_enthalpy_newton(self, enthalpy, t_min=self.t_min, t_max=self.t_max)

    def potential_temperature(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        target_psi = self.psi(t) - self.gas_constant * np.log(p / self.p_ref)
        return invert_psi_newton(self, target_psi, t_min=self.t_min, t_max=self.t_max)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "provider": "NASAThermo",
            "species": self.species,
            "molar_mass": self.molar_mass,
            "gas_constant": self.gas_constant,
            "source": self.source,
            "cea_note": self.cea_note,
            "t_ref": self.t_ref,
            "p_ref": self.p_ref,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "intervals": [
                {"t_min": t0, "t_max": t1, "coefficients": list(c)}
                for t0, t1, c in self.intervals
            ],
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class MixtureThermo:
    """Fixed mole-fraction mixture of ThermoProvider species."""

    species: tuple[ThermoProvider, ...]
    mole_fractions: tuple[float, ...]
    t_ref: float = T_REF
    p_ref: float = P_REF

    def __post_init__(self) -> None:
        if len(self.species) != len(self.mole_fractions):
            raise ValueError("species and mole_fractions length mismatch")
        if len(self.species) < 1:
            raise ValueError("mixture requires at least one species")
        total = float(sum(self.mole_fractions))
        if not np.isfinite(total) or abs(total - 1.0) > 1.0e-12:
            raise ValueError("mole fractions must sum to 1")
        if any(x < 0.0 for x in self.mole_fractions):
            raise ValueError("mole fractions must be nonnegative")
        positive("t_ref", self.t_ref)
        positive("p_ref", self.p_ref)

    @property
    def molar_mass(self) -> float:
        return float(
            sum(x * sp.molar_mass for sp, x in zip(self.species, self.mole_fractions))
        )

    @property
    def gas_constant(self) -> float:
        return R_UNIVERSAL / self.molar_mass

    @property
    def t_min(self) -> float:
        return max(float(getattr(sp, "t_min", 200.0)) for sp in self.species)

    @property
    def t_max(self) -> float:
        return min(float(getattr(sp, "t_max", 6000.0)) for sp in self.species)

    def specific_heat(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        # cp_mix = (Σ x_s cp_s^molar) / μ_mix ; cp_s^molar = cp_s * μ_s
        numerator = np.zeros(t.shape, dtype=float)
        for sp, x in zip(self.species, self.mole_fractions):
            numerator = numerator + x * sp.specific_heat(t) * sp.molar_mass
        return numerator / self.molar_mass

    def enthalpy(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        # Providers already satisfy h(T_ref)=0; do not subtract again.
        numerator = np.zeros(t.shape, dtype=float)
        for sp, x in zip(self.species, self.mole_fractions):
            numerator = numerator + x * sp.enthalpy(t) * sp.molar_mass
        return numerator / self.molar_mass

    def psi(self, temperature: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        numerator = np.zeros(t.shape, dtype=float)
        for sp, x in zip(self.species, self.mole_fractions):
            numerator = numerator + x * sp.psi(t) * sp.molar_mass
        return numerator / self.molar_mass

    def entropy(self, temperature: ArrayLike, pressure: ArrayLike) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        if np.any(p <= 0.0):
            raise ValueError("pressure must be positive")
        return self.psi(t) - self.gas_constant * np.log(p / self.p_ref)

    def nabla_ad_at(self, temperature: ArrayLike) -> NDArray[np.float64]:
        return self.gas_constant / self.specific_heat(temperature)

    def density(self, pressure: ArrayLike, temperature: ArrayLike) -> NDArray[np.float64]:
        return _density_ideal(self.gas_constant, pressure, temperature)

    def invert_enthalpy(self, enthalpy: ArrayLike) -> NDArray[np.float64]:
        return invert_enthalpy_newton(
            self, enthalpy, t_min=self.t_min, t_max=self.t_max
        )

    def potential_temperature(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> NDArray[np.float64]:
        t = _as_float_array("temperature", temperature)
        p = _as_float_array("pressure", pressure)
        _match_shapes(t, p, "temperature/pressure")
        target_psi = self.psi(t) - self.gas_constant * np.log(p / self.p_ref)
        return invert_psi_newton(
            self, target_psi, t_min=self.t_min, t_max=self.t_max
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "provider": "MixtureThermo",
            "molar_mass": self.molar_mass,
            "gas_constant": self.gas_constant,
            "mole_fractions": list(self.mole_fractions),
            "species": [sp.as_metadata() for sp in self.species],
            "t_ref": self.t_ref,
            "p_ref": self.p_ref,
        }


def h2_he_mixture(x_he: float, **kwargs: Any) -> MixtureThermo:
    """Fixed-composition H2/He mixture with NASA H2 and exact monatomic He."""
    positive_fraction = float(x_he)
    if positive_fraction < 0.0 or positive_fraction > 1.0:
        raise ValueError("x_he must lie in [0, 1]")
    return MixtureThermo(
        species=(NASAThermo.from_json(), monatomic_helium()),
        mole_fractions=(1.0 - positive_fraction, positive_fraction),
        **kwargs,
    )

import json
from pathlib import Path

from convection_mlt.config import PhysicsConfig
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import solve_adaptive


FIXTURES = Path(__file__).parents[1] / "fixtures"


def run_fixture(name: str):
    case = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    physics = PhysicsConfig(
        gravity=case["gravity_m_s2"], alpha=case["alpha"]
    )
    grid = build_grid(
        log_pressure_edges(
            case["p_bottom_pa"],
            case["p_top_pa"],
            case["n_layers"],
        ),
        physics.gravity,
    )
    temperature = case["temperature_reference_k"] * (
        grid.pressure_centres / case["pressure_reference_pa"]
    ) ** case["power_law_exponent"]
    return case, solve_adaptive(grid, temperature, physics)


def test_tiny_stable_fixture_is_deterministic():
    case, result = run_fixture("tiny_stable.json")
    assert result.status.value == case["expected_status"]


def test_tiny_adiabat_fixture_is_deterministic():
    case, result = run_fixture("tiny_adiabat.json")
    assert result.status.value == case["expected_status"]

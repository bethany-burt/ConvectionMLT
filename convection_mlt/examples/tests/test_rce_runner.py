"""Interface tests for the examples/rce user-facing runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2]
EXAMPLES = PKG / "examples"
RCE = EXAMPLES / "rce"
RUNNER = RCE / "run_rce.py"
SRC = PKG / "src"
EXAMPLE = RCE / "example_config.json"
PYTHON = sys.executable


def _run_cli(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env_pythonpath = str(SRC)
    return subprocess.run(
        [PYTHON, str(RUNNER), *args],
        cwd=str(cwd or PKG),
        capture_output=True,
        text=True,
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "PYTHONPATH": env_pythonpath,
        },
        check=False,
    )


def test_invalid_n48_rejected(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    cfg["physics"]["n_layers"] = 48
    cfg["solve"]["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "bad_n.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2
    assert "INVALID INPUT" in proc.stdout or "INVALID INPUT" in proc.stderr


def test_gate_cannot_be_loosened(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    cfg["solve"]["gate"] = 0.01
    cfg["solve"]["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "loose_gate.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2
    assert "gate" in (proc.stdout + proc.stderr).lower()


def test_isothermal_seed_rejected(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    cfg["physics"]["seed"] = "isothermal"
    cfg["solve"]["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "iso.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2


def test_unknown_key_rejected(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    cfg["physics"]["magic"] = 1
    path = tmp_path / "unk.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2


def test_validate_config_unit() -> None:
    sys.path.insert(0, str(RCE))
    from config import ConfigError, default_example_dict, validate_config

    ok = validate_config(default_example_dict())
    assert ok.gate == 0.001
    assert ok.envelope_status == "INSIDE"
    with pytest.raises(ConfigError):
        validate_config({"physics": {"n_layers": 48}, "solve": {"output_dir": "x"}})


@pytest.mark.slow
def test_rc_seed_production_converged(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    out = tmp_path / "rc_out"
    cfg["physics"]["seed"] = "radiative_convective"
    cfg["solve"]["output_dir"] = str(out)
    path = tmp_path / "rc.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path), "--force"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CONVERGED" in proc.stdout
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "CONVERGED"
    assert status["topology_ok"] is True
    assert status["convergence"] is True
    centres = (out / "profiles_centres.csv").read_text().strip().splitlines()
    interfaces = (out / "profiles_interfaces.csv").read_text().strip().splitlines()
    assert len(centres) - 1 == 96
    assert len(interfaces) - 1 == 97
    assert (out / "figure_summary.png").is_file()
    assert (out / "convergence.csv").is_file()
    assert (out / "run.log").is_file()
    conv = (out / "convergence.csv").read_text()
    assert "phase" in conv.splitlines()[0]
    assert "pseudo_time_s" in conv.splitlines()[0]


@pytest.mark.slow
def test_re_seed_production_converged(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    out = tmp_path / "re_out"
    cfg["physics"]["seed"] = "radiative_equilibrium"
    cfg["solve"]["output_dir"] = str(out)
    path = tmp_path / "re.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path), "--force"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "CONVERGED"
    assert status["topology_ok"] is True


@pytest.mark.slow
def test_under_resourced_not_converged(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    out = tmp_path / "fail_out"
    cfg["physics"]["seed"] = "radiative_equilibrium"
    cfg["solve"]["output_dir"] = str(out)
    cfg["advanced"]["max_steps_live_polish"] = 1
    cfg["advanced"]["max_recovery_cycles"] = 0
    cfg["advanced"]["max_steps_continuation"] = 1
    path = tmp_path / "under.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path), "--force"])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "NOT CONVERGED" in proc.stdout
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "NOT CONVERGED"


def test_refuse_nonempty_without_force(tmp_path: Path) -> None:
    cfg = json.loads(EXAMPLE.read_text())
    out = tmp_path / "busy"
    out.mkdir()
    (out / "marker.txt").write_text("x")
    cfg["solve"]["output_dir"] = str(out)
    cfg["advanced"]["max_recovery_cycles"] = 0
    cfg["advanced"]["max_steps_live_polish"] = 1
    path = tmp_path / "busy.json"
    path.write_text(json.dumps(cfg))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2
    assert "non-empty" in (proc.stdout + proc.stderr).lower() or "INVALID INPUT" in proc.stdout

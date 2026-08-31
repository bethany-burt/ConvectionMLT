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
EXAMPLE = RCE / "cfg_demo.py"
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


def _write_cfg(tmp_path: Path, **overrides: object) -> Path:
    lines = [EXAMPLE.read_text().rstrip(), ""]
    for key, value in overrides.items():
        if isinstance(value, str):
            lines.append(f"{key} = {value!r}")
        elif value is None:
            lines.append(f"{key} = None")
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value)}")
        else:
            lines.append(f"{key} = {value}")
    path = tmp_path / "case.py"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_invalid_n3_rejected(tmp_path: Path) -> None:
    path = _write_cfg(tmp_path, n_layers=3, output_dir=str(tmp_path / "out"))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2
    assert "n_layers" in (proc.stdout + proc.stderr).lower()


def test_gate_cannot_be_set_in_case_file(tmp_path: Path) -> None:
    text = EXAMPLE.read_text() + "\ngate = 0.01\n"
    path = tmp_path / "bad_gate.py"
    path.write_text(text)
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2
    assert "gate" in (proc.stdout + proc.stderr).lower()


def test_isothermal_seed_rejected(tmp_path: Path) -> None:
    path = _write_cfg(tmp_path, seed="isothermal", output_dir=str(tmp_path / "out"))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2


def test_invalid_x_he_rejected(tmp_path: Path) -> None:
    path = _write_cfg(tmp_path, x_he=1.5, output_dir=str(tmp_path / "out"))
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2
    assert "x_he" in (proc.stdout + proc.stderr).lower()


def test_validate_cfg_unit() -> None:
    sys.path.insert(0, str(RCE))
    from load_cfg import ConfigError, load_and_validate, validate_user_cfg

    ok = load_and_validate(EXAMPLE)
    assert ok.gate == 0.001
    assert ok.p_bottom_bar == 10.0
    assert ok.p_top_bar == 1.0e-5
    assert ok.p_bottom == 1.0e6
    assert ok.p_top == 1.0
    assert ok.envelope_status in {
        "INSIDE",
        "INSIDE_VALIDATED_ENVELOPE",
        "OUTSIDE",
        "EXPERIMENTAL_OUTSIDE_VALIDATED_ENVELOPE",
    }
    assert ok.resolved_output_dir.endswith("firr1500_alpha1_n100")

    with pytest.raises(ConfigError):
        validate_user_cfg({"n_layers": 3, "output_dir": "x"})

    snap = ok.to_snapshot()
    assert snap["atmosphere"]["nabla_ad_at_1500K"] > 0.0
    assert snap["atmosphere"]["cp_at_1500K"] > 0.0


def test_arbitrary_n_layers_loads(tmp_path: Path) -> None:
    sys.path.insert(0, str(RCE))
    from load_cfg import load_and_validate

    path = _write_cfg(tmp_path, n_layers=100, output_dir=str(tmp_path / "out100"))
    cfg = load_and_validate(path)
    assert cfg.n_layers == 100
    assert cfg.envelope_status != "INSIDE_VALIDATED_ENVELOPE"
    assert any("n_layers=100" in w for w in cfg.envelope_warnings)


@pytest.mark.slow
def test_rc_seed_production_converged(tmp_path: Path) -> None:
    out = tmp_path / "rc_out"
    path = _write_cfg(
        tmp_path,
        seed="radiative_convective",
        output_dir=str(out),
        out_name="",
    )
    proc = _run_cli(["--config", str(path), "--force"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CONVERGED" in proc.stdout
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "CONVERGED"
    assert status["topology_ok"] is True
    assert status["convergence"] is True
    assert (out / "input_cfg.py").is_file()
    assert (out / "input_resolved.json").is_file()
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
    out = tmp_path / "re_out"
    path = _write_cfg(
        tmp_path,
        seed="radiative_equilibrium",
        output_dir=str(out),
        out_name="",
    )
    proc = _run_cli(["--config", str(path), "--force"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "CONVERGED"
    assert status["topology_ok"] is True


@pytest.mark.slow
def test_under_resourced_not_converged(tmp_path: Path) -> None:
    out = tmp_path / "fail_out"
    path = _write_cfg(
        tmp_path,
        seed="radiative_equilibrium",
        output_dir=str(out),
        out_name="",
        max_steps_live_polish=1,
        max_recovery_cycles=0,
        max_steps_continuation=1,
    )
    proc = _run_cli(["--config", str(path), "--force"])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "NOT CONVERGED" in proc.stdout
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "NOT CONVERGED"


def test_refuse_nonempty_without_force(tmp_path: Path) -> None:
    out = tmp_path / "busy"
    out.mkdir()
    (out / "marker.txt").write_text("x")
    path = _write_cfg(
        tmp_path,
        output_dir=str(out),
        out_name="",
        max_recovery_cycles=0,
        max_steps_live_polish=1,
    )
    proc = _run_cli(["--config", str(path)])
    assert proc.returncode == 2
    assert "non-empty" in (proc.stdout + proc.stderr).lower() or "INVALID INPUT" in proc.stdout

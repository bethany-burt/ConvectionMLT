"""Shared pytest configuration for convection_mlt."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src layout resolves without manual PYTHONPATH in local runs.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: full production RCE solve (seconds to minutes); skip with -m 'not slow'",
    )

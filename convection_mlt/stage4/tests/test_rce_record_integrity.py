"""Stored RCE records: checksums and simulated_time == Σ history.dt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
RESULTS = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(EXPERIMENTS))

from rce_record import (  # noqa: E402
    finalize_record,
    history_is_complete_clock,
    history_simulated_time,
    merge_continuation,
    verify_record_checksums,
)


def _stored_records():
    records = []
    n192 = RESULTS / "n192_implicit_rce.json"
    if n192.exists():
        records.append(("n192", json.loads(n192.read_text())))
    n384 = RESULTS / "n384_implicit_rce.json"
    if n384.exists():
        records.append(("n384", json.loads(n384.read_text())))
    nested = RESULTS / "nested_rce_family.json"
    if nested.exists():
        members = (json.loads(nested.read_text()).get("members") or {})
        for key, rec in members.items():
            if isinstance(rec, dict) and rec.get("history"):
                records.append((f"nested_{key}", rec))
    return records


_RECORDS = _stored_records()


@pytest.mark.parametrize(
    "name,record",
    _RECORDS,
    ids=[name for name, _record in _RECORDS] or ["no-records"],
)
def test_stored_rce_record_checksums_and_simulated_time(name, record):
    verify_record_checksums(record)
    dts = (record.get("history") or {}).get("dt") or []
    if dts and history_is_complete_clock(record):
        assert record["simulated_time"] == pytest.approx(history_simulated_time(record), rel=0.0, abs=1e-6)


def test_n384_polish_segment_keeps_inherited_clock():
    path = RESULTS / "n384_implicit_rce.json"
    if not path.exists():
        pytest.skip("n384_implicit_rce.json not stored")
    record = json.loads(path.read_text())
    verify_record_checksums(record)
    dts = (record.get("history") or {}).get("dt") or []
    assert dts
    assert history_is_complete_clock(record) is False
    assert record["simulated_time"] > history_simulated_time(record)
    assert record.get("profile_checksum_sha256") == (
        "5e0bd359a7e352863c767faf5441d93968370d358a5b0270c8f6ebf293d875ec"
    )


def test_merge_continuation_uses_chunk_absolute_time():
    base = {
        "steps_accepted": 10,
        "rejections": 0,
        "simulated_time": 100.0,
        "wall_time_s": 1.0,
        "history": {"dt": [10.0] * 10, "flux_flatness": [0.01] * 10},
        "rejection_reasons": [],
        "flux_total": [300.0, 300.0],
        "flux_rad": [300.0, 300.0],
        "flux_conv": [0.0, 0.0],
        "mass_path": [1.0],
        "f_int": 300.0,
        "temperature": [1000.0],
        "pressure_centres": [1.0e5],
        "enthalpy": [1.0e7],
    }
    chunk = {
        "steps_accepted": 2,
        "rejections": 0,
        "simulated_time": 130.0,
        "wall_time_s": 0.5,
        "history": {"dt": [15.0, 15.0], "flux_flatness": [0.009, 0.008]},
        "rejection_reasons": [],
        "flux_total": [300.0, 300.0],
        "flux_rad": [300.0, 300.0],
        "flux_conv": [0.0, 0.0],
        "mass_path": [1.0],
        "f_int": 300.0,
        "temperature": [1001.0],
        "pressure_centres": [1.0e5],
        "enthalpy": [1.0e7],
        "environment": {},
    }
    merged = merge_continuation(base, chunk)
    assert merged["simulated_time"] == 130.0
    assert merged["steps_accepted"] == 12
    assert history_simulated_time(merged) == pytest.approx(130.0)
    verify_record_checksums(merged)


def test_finalize_record_after_metadata_mutation():
    rec = {
        "flux_total": [1.0, 1.0],
        "flux_rad": [1.0, 1.0],
        "flux_conv": [0.0, 0.0],
        "mass_path": [1.0],
        "f_int": 1.0,
        "temperature": [800.0],
        "pressure_centres": [1.0e5],
        "enthalpy": [1.0],
        "note": "before",
    }
    finalize_record(rec)
    first = rec["record_checksum_sha256"]
    rec["note"] = "after"
    assert rec["record_checksum_sha256"] == first
    finalize_record(rec)
    assert rec["record_checksum_sha256"] != first
    verify_record_checksums(rec)

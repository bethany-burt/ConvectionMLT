import json

import numpy as np

from convection_mlt.diagnostics import mixing_timescales
from convection_mlt.metadata import json_safe, run_metadata


def test_inactive_timescales_use_inf_mask_and_json_null():
    turn, mix, active = mixing_timescales(
        [0.0, 2.0], [0.0, 4.0], [0.0, 10.0], [0.0, 8.0]
    )
    assert np.isinf(turn[0]) and np.isinf(mix[0])
    assert not active[0]
    assert turn[1] == 0.5
    assert mix[1] == 12.5
    assert active[1]
    encoded = json.dumps(json_safe({"turn": turn, "mix": mix}), allow_nan=False)
    assert "null" in encoded
    assert json.dumps(json_safe({"active": np.array([True, False])})) == (
        '{"active": [true, false]}'
    )


def test_metadata_records_actual_closure_units_and_sources():
    metadata = run_metadata(
        {
            "physics": {
                "alpha": 1.0,
                "closure_prefactor": 0.25,
                "gravity": 15.0,
            }
        }
    )
    assert metadata["closure"]["prefactor"] == 0.25
    assert metadata["closure"]["sources"]
    assert metadata["units"]["flux"] == "W m^-2"
    assert metadata["git_commit"]
    assert metadata["git_dirty"] in (True, False, None)

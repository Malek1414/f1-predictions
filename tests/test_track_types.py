import logging

import pytest

from f1pred.data.track_types import TRACK_TYPES, load_track_types, track_type_for


def test_mapping_covers_sample_circuits(driver_race):
    mapping = load_track_types()
    missing = set(driver_race.circuit_id) - set(mapping)
    assert not missing
    assert set(mapping.values()) <= set(TRACK_TYPES)
    assert mapping["monaco"] == "street" and mapping["monza"] == "high_speed"


def test_unknown_circuit_falls_back_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert track_type_for("moon", {"monaco": "street"}) == "mixed"
    assert "moon" in caplog.text


def test_invalid_value_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("monaco: oval\n")
    with pytest.raises(ValueError, match="oval"):
        load_track_types(p)

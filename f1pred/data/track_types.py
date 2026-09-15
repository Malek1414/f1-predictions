"""Circuit type lookup from the hand-curated YAML. Spec 3.3."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

TRACK_TYPES = ("street", "high_speed", "high_downforce", "mixed")
DEFAULT_TRACK_TYPE = "mixed"
DEFAULT_PATH = Path(__file__).parent / "track_types.yaml"
_warned: set[str] = set()


def load_track_types(path: Path | None = None) -> dict[str, str]:
    data = yaml.safe_load(Path(path or DEFAULT_PATH).read_text()) or {}
    for circuit, kind in data.items():
        if kind not in TRACK_TYPES:
            raise ValueError(
                f"track type for {circuit!r} is {kind!r}, expected one of {TRACK_TYPES}"
            )
    return {str(k): str(v) for k, v in data.items()}


def track_type_for(circuit_id: str, mapping: Mapping[str, str]) -> str:
    kind = mapping.get(circuit_id)
    if kind is None:
        if circuit_id not in _warned:
            log.warning("circuit %r has no track type; using %s", circuit_id, DEFAULT_TRACK_TYPE)
            _warned.add(circuit_id)
        return DEFAULT_TRACK_TYPE
    return kind

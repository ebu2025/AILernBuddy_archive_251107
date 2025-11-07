import json
from pathlib import Path

import pytest

from bloom_levels import BLOOM_LEVELS, BloomLevelConfigError, BloomLevelRegistry


def test_default_registry_exposes_sequence():
    sequence = BLOOM_LEVELS.sequence()
    assert sequence[0] == BLOOM_LEVELS.lowest_level()
    assert len(sequence) >= 3


def test_custom_registry_validates_and_sorts(tmp_path: Path):
    data = [
        {"id": "L1", "label": "Level 1", "description": "Intro", "min_score": 0.2},
        {"id": "L2", "label": "Level 2", "description": "Advance", "min_score": 0.7},
    ]
    cfg = tmp_path / "levels.json"
    cfg.write_text(json.dumps(data), encoding="utf-8")

    registry = BloomLevelRegistry(cfg)
    assert registry.sequence() == ("L1", "L2")
    assert registry.thresholds_descending()[0] == (0.7, "L2")
    assert registry.k_level_sequence(1) == ("L1",)
    assert registry.get("L2").label == "Level 2"

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps([{ "label": "Missing id" }]), encoding="utf-8")
    with pytest.raises(BloomLevelConfigError):
        BloomLevelRegistry(broken)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps([{ "id": "L1", "label": "Bad", "min_score": 2 }]), encoding="utf-8")
    with pytest.raises(BloomLevelConfigError):
        BloomLevelRegistry(invalid)

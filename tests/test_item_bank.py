import json
from pathlib import Path

import pytest

import db
from item_bank import ItemBank, ItemValidationError


@pytest.fixture
def sample_items(tmp_path: Path) -> Path:
    items = [
        {
            "id": "math-basic-001",
            "domain": "math",
            "skill_id": "math.arithmetic.addition",
            "bloom_level": "K1",
            "stimulus": "What is 1+1?",
            "answer_key": "2",
            "rubric_id": None,
            "difficulty": -0.5,
            "elo_target": 950.0,
            "metadata": {"tags": ["addition"]},
            "exposure_limit": 1,
            "references": [],
        },
        {
            "id": "math-basic-002",
            "domain": "math",
            "skill_id": "math.arithmetic.addition",
            "bloom_level": "K1",
            "stimulus": "What is 2+2?",
            "answer_key": "4",
            "rubric_id": None,
            "difficulty": 0.0,
            "elo_target": 980.0,
            "metadata": {"tags": ["addition"]},
            "exposure_limit": 3,
            "references": [],
        },
    ]
    path = tmp_path / "items.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_item_bank_loads_and_syncs(temp_db, sample_items: Path):
    bank = ItemBank(sample_items)
    assert len(bank.items) == 2

    rows = db.list_item_bank(domain="math")
    assert {row["id"] for row in rows} == {"math-basic-001", "math-basic-002"}


def test_item_bank_exposure_limit(temp_db, sample_items: Path):
    bank = ItemBank(sample_items)
    first_pick = bank.select_items(domain="math", target_difficulty=-0.5)[0]
    bank.mark_served([first_pick["id"]])

    next_pick = bank.select_items(domain="math", target_difficulty=-0.5)[0]
    assert next_pick["id"] != first_pick["id"]


def test_item_bank_validation(tmp_path: Path):
    bad_items = [{"id": "bad"}]
    path = tmp_path / "bad_items.json"
    path.write_text(json.dumps(bad_items), encoding="utf-8")

    with pytest.raises(ItemValidationError):
        ItemBank(path, auto_sync=False)


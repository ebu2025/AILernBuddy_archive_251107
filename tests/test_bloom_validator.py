"""Regression tests for the Bloom coverage validator."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from item_bank import ItemBank
from scripts import validate_bloom_coverage
from scripts.bloom_validate import (
    BloomStemClassifier,
    DEFAULT_HIGH_LEVELS,
    compute_coverage_report,
)

BASELINE_PATH = Path(__file__).parent / "baselines" / "bloom_coverage.json"
MIN_HIGH_LEVEL_RATIO = 0.3
DRIFT_TOLERANCE = 0.02


def _load_report() -> dict:
    bank = ItemBank(auto_sync=False)
    classifier = BloomStemClassifier()
    return compute_coverage_report(bank.items, classifier)


def test_high_level_coverage_meets_threshold() -> None:
    report = _load_report()
    missing_domains = [domain for domain in report["domains"] if report["domains"][domain]["total_items"] == 0]
    assert not missing_domains, f"Validator report contains empty domains: {missing_domains}"

    for domain, stats in report["domains"].items():
        ratio = stats["actual_high_ratio"]
        assert ratio >= MIN_HIGH_LEVEL_RATIO, (
            f"Domain '{domain}' Bloom K5–K6 coverage {ratio:.2%} fell below the {MIN_HIGH_LEVEL_RATIO:.0%} threshold."
        )


def test_bloom_coverage_matches_baseline() -> None:
    if not BASELINE_PATH.exists():
        pytest.skip("Bloom coverage baseline missing; run scripts/bloom_validate.py --output to create it.")

    report = _load_report()
    with BASELINE_PATH.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    assert report["high_levels"] == baseline["high_levels"] == list(DEFAULT_HIGH_LEVELS)

    baseline_domains = baseline["domains"]
    report_domains = report["domains"]
    assert set(report_domains) == set(baseline_domains), (
        "Domain set drift detected. Update tests/baselines/bloom_coverage.json if domains were intentionally added or removed."
    )

    for domain, baseline_stats in baseline_domains.items():
        stats = report_domains[domain]
        assert stats["actual_counts"] == baseline_stats["actual_counts"], (
            "Bloom level distribution drift detected for domain "
            f"'{domain}'. Update the baseline if the changes are expected."
        )
        assert stats["total_items"] == baseline_stats["total_items"]
        diff = abs(stats["actual_high_ratio"] - baseline_stats["actual_high_ratio"])
        assert diff <= DRIFT_TOLERANCE, (
            f"Domain '{domain}' high-level coverage drifted by {diff:.2%} which exceeds the {DRIFT_TOLERANCE:.0%} tolerance."
        )

    assert report["global"]["total_items"] == baseline["global"]["total_items"]
    assert report["global"]["actual_high_count"] == baseline["global"]["actual_high_count"]


def _extract_level_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if re.match(r"^K[1-6]:", line)]


def test_validator_prints_all_bloom_levels(temp_db, capsys):
    exit_code = validate_bloom_coverage.main([])
    captured = capsys.readouterr()

    assert exit_code == 0

    level_lines = _extract_level_lines(captured.out)
    assert len(level_lines) == 6
    assert [line.split(":", 1)[0] for line in level_lines] == [f"K{i}" for i in range(1, 7)]

    for line in level_lines:
        assert re.match(r"^K[1-6]: \d{1,3}\.\d{2}%$", line)


def test_validator_enforces_k6_threshold(temp_db, capsys, monkeypatch):
    monkeypatch.setenv(validate_bloom_coverage.K6_ENV_VAR, "0.95")

    exit_code = validate_bloom_coverage.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "K6 share" in captured.err
    assert "  K6 share: 95.00%" in captured.out

    k6_line = next(line for line in _extract_level_lines(captured.out) if line.startswith("K6:"))
    assert "(threshold: 95.00%)" in k6_line

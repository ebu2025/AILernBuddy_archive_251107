import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validate_bloom_coverage


def test_validator_passes_with_defaults(temp_db, capsys):
    exit_code = validate_bloom_coverage.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = captured.out.strip().splitlines()
    assert lines[:5] == [
        "Effective thresholds:",
        "  Global high-level coverage: 15.00%",
        "  Domain high-level coverage: 10.00%",
        "  Classifier predicted coverage: not enforced",
        "  K6 share: not enforced",
    ]
    percentage_lines = [line for line in lines if re.match(r"^K[1-6]: ", line)]
    assert len(percentage_lines) == 6
    expected_prefixes = [f"K{i}:" for i in range(1, 7)]
    assert [f"{line.split(':', 1)[0]}:" for line in percentage_lines] == expected_prefixes
    for line in percentage_lines:
        assert re.match(r"^K[1-6]: \d{1,3}\.\d{2}%$", line)
    assert any(line.strip().startswith('"global"') for line in lines)


def test_validator_fails_with_high_threshold(temp_db, capsys):
    exit_code = validate_bloom_coverage.main(["--high-level-threshold", "0.95"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "fell below" in captured.err
    assert "  Global high-level coverage: 95.00%" in captured.out
    percentage_lines = [
        line for line in captured.out.splitlines() if re.match(r"^K[1-6]: \d{1,3}\.\d{2}%$", line)
    ]
    assert len(percentage_lines) == 6
    for line in percentage_lines:
        assert re.match(r"^K[1-6]: \d{1,3}\.\d{2}%$", line)


def test_validator_passes_with_k6_share_threshold(temp_db, capsys, monkeypatch):
    monkeypatch.setenv("K6_MIN_SHARE", "0.01")
    exit_code = validate_bloom_coverage.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "  K6 share: 1.00%" in captured.out
    k6_line = next(line for line in captured.out.splitlines() if line.startswith("K6:"))
    assert "(threshold: 1.00%)" in k6_line
    assert "K6 share" not in captured.err


def test_validator_fails_when_k6_share_below_threshold(temp_db, capsys, monkeypatch):
    monkeypatch.setenv("K6_MIN_SHARE", "0.95")
    exit_code = validate_bloom_coverage.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "K6 share" in captured.err
    assert "  K6 share: 95.00%" in captured.out
    k6_line = next(line for line in captured.out.splitlines() if line.startswith("K6:"))
    assert "(threshold: 95.00%)" in k6_line


def test_env_overrides_global_threshold(temp_db, capsys, monkeypatch):
    monkeypatch.setenv("BLOOM_HIGH_MIN", "0.5")
    exit_code = validate_bloom_coverage.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "  Global high-level coverage: 50.00%" in captured.out
    assert "50% threshold" in captured.err


def test_env_overrides_domain_threshold(temp_db, capsys, monkeypatch):
    monkeypatch.setenv("BLOOM_DOMAIN_MIN", "0.5")
    exit_code = validate_bloom_coverage.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "  Domain high-level coverage: 50.00%" in captured.out
    assert "50% threshold" in captured.err


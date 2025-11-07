"""Offline Bloom coverage validator that enforces K5–K6 coverage thresholds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_bank import ItemBank
from scripts.bloom_validate import BloomStemClassifier, DEFAULT_HIGH_LEVELS, compute_coverage_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=str,
        default="items.json",
        help="Path to the item bank JSON file (default: items.json)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Optional JSON keyword classifier configuration",
    )
    parser.add_argument(
        "--high-level-threshold",
        type=float,
        default=0.15,
        help="Minimum required global ratio of high-level (K5/K6) coverage",
    )
    parser.add_argument(
        "--min-domain-ratio",
        type=float,
        default=0.1,
        help="Minimum required high-level coverage per domain",
    )
    parser.add_argument(
        "--predicted-threshold",
        type=float,
        default=None,
        help="Optional threshold for classifier-predicted high-level coverage",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON report",
    )
    return parser


LEVELS: tuple[str, ...] = ("K1", "K2", "K3", "K4", "K5", "K6")
BLOOM_HIGH_ENV_VAR = "BLOOM_HIGH_MIN"
BLOOM_DOMAIN_ENV_VAR = "BLOOM_DOMAIN_MIN"
K6_ENV_VAR = "K6_MIN_SHARE"


def _write_output(report: dict, output_path: str | None) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if output_path:
        Path(output_path).write_text(payload + "\n", encoding="utf-8")
    print(payload)


def _threshold_failed(value: float, threshold: float | None) -> bool:
    return threshold is not None and value < threshold


def _compute_level_percentages(
    domain_stats: Mapping[str, Mapping[str, object]],
    *,
    levels: Iterable[str] = LEVELS,
) -> dict[str, float]:
    totals = {level: 0 for level in levels}
    total_items = 0

    for stats in domain_stats.values():
        counts = stats.get("actual_counts") or {}
        if not isinstance(counts, Mapping):
            continue
        for level in levels:
            totals[level] += int(counts.get(level, 0))
        total_items += int(stats.get("total_items", 0))

    if total_items == 0:
        return {level: 0.0 for level in levels}

    return {level: totals[level] / total_items for level in levels}


def _parse_env_threshold(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    try:
        threshold = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float between 0 and 1, got {value!r}.") from exc

    if not 0 <= threshold <= 1:
        raise ValueError(f"{name} must be between 0 and 1 inclusive, got {threshold}.")

    return threshold


def _print_level_percentages(
    percentages: Mapping[str, float], *, k6_threshold: float | None = None
) -> None:
    for level in LEVELS:
        ratio = percentages.get(level, 0.0)
        suffix = ""
        if level == "K6" and k6_threshold is not None:
            suffix = f" (threshold: {k6_threshold:.2%})"
        print(f"{level}: {ratio:.2%}{suffix}")


def _print_thresholds(
    *,
    global_threshold: float,
    domain_threshold: float,
    predicted_threshold: float | None,
    k6_threshold: float | None,
) -> None:
    print("Effective thresholds:")
    print(f"  Global high-level coverage: {global_threshold:.2%}")
    print(f"  Domain high-level coverage: {domain_threshold:.2%}")
    if predicted_threshold is None:
        print("  Classifier predicted coverage: not enforced")
    else:
        print(f"  Classifier predicted coverage: {predicted_threshold:.2%}")
    if k6_threshold is None:
        print("  K6 share: not enforced")
    else:
        print(f"  K6 share: {k6_threshold:.2%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    bank = ItemBank(args.items, auto_sync=False)
    classifier = BloomStemClassifier.from_path(args.classifier)
    high_levels = DEFAULT_HIGH_LEVELS
    report = compute_coverage_report(bank.items, classifier, high_levels=high_levels)

    try:
        env_high_threshold = _parse_env_threshold(BLOOM_HIGH_ENV_VAR)
        env_domain_threshold = _parse_env_threshold(BLOOM_DOMAIN_ENV_VAR)
        k6_threshold = _parse_env_threshold(K6_ENV_VAR)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    global_threshold = (
        env_high_threshold if env_high_threshold is not None else args.high_level_threshold
    )
    domain_threshold = (
        env_domain_threshold if env_domain_threshold is not None else args.min_domain_ratio
    )

    failures: list[str] = []
    global_stats = report["global"]
    actual_ratio = float(global_stats.get("actual_high_ratio") or 0.0)
    predicted_ratio = float(global_stats.get("predicted_high_ratio") or 0.0)
    domain_percentages = _compute_level_percentages(report["domains"])
    k6_ratio = float(domain_percentages.get("K6", 0.0))

    if _threshold_failed(actual_ratio, global_threshold):
        failures.append(
            f"Global high-level coverage {actual_ratio:.2%} fell below the {global_threshold:.0%} threshold."
        )
    if _threshold_failed(predicted_ratio, args.predicted_threshold):
        failures.append(
            f"Classifier high-level coverage {predicted_ratio:.2%} fell below the {args.predicted_threshold:.0%} threshold."
        )

    for domain, stats in report["domains"].items():
        ratio = float(stats.get("actual_high_ratio") or 0.0)
        if _threshold_failed(ratio, domain_threshold):
            failures.append(
                f"Domain '{domain}' high-level coverage {ratio:.2%} fell below the {domain_threshold:.0%} threshold."
            )

    if _threshold_failed(k6_ratio, k6_threshold):
        failures.append(
            f"K6 share {k6_ratio:.2%} fell below the {K6_ENV_VAR} {k6_threshold:.0%} threshold."
        )

    if failures:
        for message in failures:
            print(message, file=sys.stderr)

    _print_thresholds(
        global_threshold=global_threshold,
        domain_threshold=domain_threshold,
        predicted_threshold=args.predicted_threshold,
        k6_threshold=k6_threshold,
    )
    _print_level_percentages(domain_percentages, k6_threshold=k6_threshold)
    _write_output(report, args.output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


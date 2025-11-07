"""Validate Bloom-level coverage across the item bank."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_bank import ItemBank

DEFAULT_RULES: List[Tuple[str, Sequence[str]]] = [
    ("K6", ("design", "compose", "formulate", "innovate", "create")),
    ("K5", ("evaluate", "justify", "critique", "analyze", "defend")),
    ("K4", ("compare", "organize", "differentiate", "outline")),
    ("K3", ("apply", "solve", "execute", "demonstrate")),
    ("K2", ("explain", "summarize", "classify", "describe")),
    ("K1", ("define", "identify", "recall", "list")),
]
DEFAULT_FALLBACK_LEVEL = "K3"
DEFAULT_HIGH_LEVELS = ("K5", "K6")


class BloomStemClassifier:
    """Very small heuristic classifier for Bloom levels based on keywords."""

    def __init__(
        self,
        rules: Mapping[str, Sequence[str]] | Sequence[Tuple[str, Sequence[str]]] | None = None,
        *,
        fallback_level: str = DEFAULT_FALLBACK_LEVEL,
    ) -> None:
        if rules is None:
            rules = DEFAULT_RULES
        if isinstance(rules, Mapping):
            ordered_rules = list(rules.items())
        else:
            ordered_rules = list(rules)
        if not ordered_rules:
            raise ValueError("At least one rule must be provided for the classifier")
        self._rules: List[Tuple[str, Tuple[str, ...]]] = [
            (str(level), tuple(str(keyword).lower() for keyword in keywords))
            for level, keywords in ordered_rules
        ]
        self._fallback = str(fallback_level)

    @property
    def fallback_level(self) -> str:
        return self._fallback

    def predict(self, stimulus: str) -> str:
        text = (stimulus or "").lower()
        for level, keywords in self._rules:
            if any(keyword in text for keyword in keywords):
                return level
        return self._fallback

    @classmethod
    def from_path(cls, path: str | Path | None) -> "BloomStemClassifier":
        if path is None:
            return cls()
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Classifier configuration not found: {path_obj}")
        with path_obj.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, Mapping):
            raise ValueError("Classifier configuration must be a JSON object")
        fallback = data.get("_fallback") or data.get("fallback") or DEFAULT_FALLBACK_LEVEL
        rules_data = data.get("rules") if "rules" in data else {k: v for k, v in data.items() if not k.startswith("_")}
        if not isinstance(rules_data, Mapping):
            raise ValueError("Classifier configuration must contain a 'rules' object")
        rules: Dict[str, List[str]] = {}
        for level, keywords in rules_data.items():
            if not isinstance(keywords, (list, tuple)):
                raise ValueError(f"Classifier rule for {level!r} must be a list of keywords")
            rules[str(level)] = [str(keyword) for keyword in keywords]
        return cls(rules, fallback_level=str(fallback))


def _normalize_high_levels(levels: Iterable[str]) -> Tuple[str, ...]:
    normalized = tuple(sorted({str(level).strip() for level in levels if str(level).strip()}))
    if not normalized:
        return DEFAULT_HIGH_LEVELS
    return normalized


def compute_domain_coverage(
    items: Sequence[Mapping[str, object]],
    classifier: BloomStemClassifier,
    *,
    high_levels: Iterable[str] = DEFAULT_HIGH_LEVELS,
) -> Dict[str, Dict[str, object]]:
    """Compute per-domain Bloom coverage statistics."""

    normalized_high = _normalize_high_levels(high_levels)
    domain_stats: Dict[str, MutableMapping[str, object]] = {}

    for item in items:
        domain = str(item.get("domain", "")).lower() or "unknown"
        stimulus = str(item.get("stimulus", ""))
        actual_level = str(item.get("bloom_level", "")).upper() or classifier.fallback_level
        predicted_level = classifier.predict(stimulus)

        stats = domain_stats.setdefault(
            domain,
            {
                "total_items": 0,
                "actual_counts": Counter(),
                "predicted_counts": Counter(),
                "mismatches": 0,
            },
        )

        stats["total_items"] += 1
        stats["actual_counts"][actual_level] += 1
        stats["predicted_counts"][predicted_level] += 1
        if actual_level != predicted_level:
            stats["mismatches"] += 1

    for domain, stats in domain_stats.items():
        total = stats["total_items"] or 0
        actual_counts: Counter = stats["actual_counts"]
        predicted_counts: Counter = stats["predicted_counts"]
        actual_high = sum(actual_counts.get(level, 0) for level in normalized_high)
        predicted_high = sum(predicted_counts.get(level, 0) for level in normalized_high)
        ratio_actual = actual_high / total if total else 0.0
        ratio_pred = predicted_high / total if total else 0.0

        stats["actual_high_count"] = actual_high
        stats["predicted_high_count"] = predicted_high
        stats["actual_high_ratio"] = ratio_actual
        stats["predicted_high_ratio"] = ratio_pred
        stats["actual_counts"] = dict(sorted(actual_counts.items()))
        stats["predicted_counts"] = dict(sorted(predicted_counts.items()))

    return {domain: dict(stats) for domain, stats in sorted(domain_stats.items())}


def compute_coverage_report(
    items: Sequence[Mapping[str, object]],
    classifier: BloomStemClassifier,
    *,
    high_levels: Iterable[str] = DEFAULT_HIGH_LEVELS,
) -> Dict[str, object]:
    """Build a JSON-serialisable coverage report for the provided items."""

    normalized_high = _normalize_high_levels(high_levels)
    domain_stats = compute_domain_coverage(items, classifier, high_levels=normalized_high)

    total_items = sum(stats["total_items"] for stats in domain_stats.values())
    total_actual_high = sum(stats["actual_high_count"] for stats in domain_stats.values())
    total_predicted_high = sum(stats["predicted_high_count"] for stats in domain_stats.values())
    total_mismatches = sum(stats["mismatches"] for stats in domain_stats.values())
    global_stats = {
        "total_items": total_items,
        "actual_high_count": total_actual_high,
        "predicted_high_count": total_predicted_high,
        "actual_high_ratio": (total_actual_high / total_items) if total_items else 0.0,
        "predicted_high_ratio": (total_predicted_high / total_items) if total_items else 0.0,
        "mismatches": total_mismatches,
    }

    return {
        "high_levels": list(normalized_high),
        "domains": domain_stats,
        "global": global_stats,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        help="Optional path to a JSON keyword classifier configuration.",
    )
    parser.add_argument(
        "--high-levels",
        type=str,
        default=",".join(DEFAULT_HIGH_LEVELS),
        help="Comma-separated Bloom levels considered 'high' (default: K5,K6)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON report instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    high_levels = tuple(level.strip().upper() for level in (args.high_levels or "").split(",") if level.strip())
    bank = ItemBank(args.items, auto_sync=False)
    classifier = BloomStemClassifier.from_path(args.classifier)
    report = compute_coverage_report(bank.items, classifier, high_levels=high_levels)
    payload = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

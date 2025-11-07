"""Report item-bank coverage and flag sparse skills/topics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

from item_bank import ItemBank


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=str,
        default="items.json",
        help="Path to the item bank JSON file (default: items.json)",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=3,
        help="Minimum number of items required per skill/topic (default: 3)",
    )
    parser.add_argument(
        "--min-bloom-coverage",
        type=int,
        default=2,
        help="Minimum number of distinct Bloom levels required per skill (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON coverage report",
    )
    return parser


def _summarise_items(items: Iterable[Dict]) -> dict:
    per_skill: Dict[str, dict] = {}
    for item in items:
        skill = item.get("skill_id", "<unknown>")
        skill_entry = per_skill.setdefault(
            skill,
            {
                "domain": item.get("domain"),
                "count": 0,
                "bloom_levels": set(),
                "elo_targets": [],
            },
        )
        skill_entry["count"] += 1
        bloom = item.get("bloom_level")
        if bloom:
            skill_entry["bloom_levels"].add(str(bloom))
        elo = item.get("elo_target")
        if isinstance(elo, (int, float)):
            skill_entry["elo_targets"].append(float(elo))
    # Convert sets to sorted lists for serialisation
    for data in per_skill.values():
        data["bloom_levels"] = sorted(data["bloom_levels"])
        if data["elo_targets"]:
            data["elo_range"] = {
                "min": min(data["elo_targets"]),
                "max": max(data["elo_targets"]),
            }
        else:
            data["elo_range"] = None
        data.pop("elo_targets", None)
    return per_skill


def _build_report(items: Sequence[Dict]) -> dict:
    per_skill = _summarise_items(items)
    totals = {
        "count": len(items),
        "skills": len(per_skill),
        "domains": sorted({item.get("domain") for item in items}),
    }
    return {"totals": totals, "skills": per_skill}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    bank = ItemBank(args.items, auto_sync=False)
    report = _build_report(bank.items)

    flagged: list[str] = []
    min_items = max(1, int(args.min_items))
    min_bloom = max(1, int(args.min_bloom_coverage))

    for skill_id, data in report["skills"].items():
        item_count = data["count"]
        bloom_levels = data["bloom_levels"]
        issues: list[str] = []
        if item_count < min_items:
            issues.append(f"only {item_count} items (min {min_items})")
        if len(bloom_levels) < min_bloom:
            issues.append(f"only {len(bloom_levels)} bloom levels (min {min_bloom})")
        if issues:
            flagged.append(f"{skill_id}: {', '.join(issues)}")

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)

    if flagged:
        for issue in flagged:
            print(f"⚠️  {issue}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

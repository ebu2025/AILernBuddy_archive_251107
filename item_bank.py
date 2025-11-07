"""Unified item bank utilities for adaptive selection and exposure control."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import db


class ItemValidationError(ValueError):
    """Raised when an item from the JSON bank fails validation."""


class ItemBank:
    """Helper for loading, validating, and selecting assessment/practice items."""

    REQUIRED_FIELDS = ("id", "domain", "skill_id", "bloom_level", "stimulus", "elo_target")

    def __init__(self, path: str | Path = "items.json", *, auto_sync: bool = True) -> None:
        self.path = Path(path)
        self._items: List[Dict[str, Any]] = []
        self._load(auto_sync=auto_sync)

    # ------------------------------------------------------------------
    # loading & validation
    # ------------------------------------------------------------------
    def _load(self, *, auto_sync: bool) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Item bank file not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if not isinstance(raw, list):
            raise ItemValidationError("Item bank root must be a JSON list")

        items: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in raw:
            if not isinstance(entry, dict):
                raise ItemValidationError("Each item must be an object")

            for field in self.REQUIRED_FIELDS:
                if field not in entry or entry[field] in (None, ""):
                    raise ItemValidationError(f"Item {entry.get('id')} missing required field '{field}'")

            item_id = str(entry["id"])
            if item_id in seen_ids:
                raise ItemValidationError(f"Duplicate item id detected: {item_id}")
            seen_ids.add(item_id)

            metadata = entry.get("metadata") or {}
            if metadata is not None and not isinstance(metadata, dict):
                raise ItemValidationError(f"Item {item_id} metadata must be an object")

            references = entry.get("references") or []
            if references is not None and not isinstance(references, list):
                raise ItemValidationError(f"Item {item_id} references must be a list")

            exposure_limit = entry.get("exposure_limit")
            if exposure_limit is not None:
                try:
                    exposure_limit = int(exposure_limit)
                except (TypeError, ValueError) as exc:
                    raise ItemValidationError(f"Item {item_id} exposure_limit must be an integer") from exc
                if exposure_limit < 0:
                    raise ItemValidationError(f"Item {item_id} exposure_limit cannot be negative")

            difficulty = entry.get("difficulty")
            if difficulty is not None:
                try:
                    difficulty = float(difficulty)
                except (TypeError, ValueError) as exc:
                    raise ItemValidationError(f"Item {item_id} difficulty must be numeric") from exc

            elo_target = entry.get("elo_target")
            try:
                elo_target = float(elo_target)
            except (TypeError, ValueError):
                raise ItemValidationError(f"Item {item_id} elo_target must be numeric")

            answer_key = entry.get("answer_key")
            rubric_id = entry.get("rubric_id")
            if not answer_key and not rubric_id:
                raise ItemValidationError(
                    f"Item {item_id} must provide an answer_key or rubric_id"
                )

            if isinstance(metadata, dict):
                choices = metadata.get("choices")
                if choices is not None:
                    if not isinstance(choices, list) or not choices:
                        raise ItemValidationError(f"Item {item_id} choices must be a non-empty list")
                    rationals = metadata.get("distractor_rationales")
                    if not isinstance(rationals, dict) or not rationals:
                        raise ItemValidationError(
                            f"Item {item_id} must include distractor_rationales for all choices"
                        )
                    missing = [
                        str(choice)
                        for choice in choices
                        if str(choice) != str(answer_key) and str(choice) not in rationals
                    ]
                    if missing:
                        raise ItemValidationError(
                            f"Item {item_id} missing distractor rationale for: {', '.join(missing)}"
                        )
                    invalid_rationales = [
                        key for key, val in rationals.items() if not isinstance(val, str) or not val.strip()
                    ]
                    if invalid_rationales:
                        raise ItemValidationError(
                            f"Item {item_id} has empty distractor rationales for: {', '.join(invalid_rationales)}"
                        )
                    metadata.setdefault("item_type", "mcq")

            items.append(
                {
                    "id": item_id,
                    "domain": str(entry["domain"]).lower(),
                    "skill_id": str(entry["skill_id"]),
                    "bloom_level": str(entry["bloom_level"]),
                    "stimulus": str(entry["stimulus"]),
                    "answer_key": answer_key,
                    "rubric_id": rubric_id,
                    "difficulty": difficulty,
                    "elo_target": elo_target,
                    "metadata": metadata,
                    "exposure_limit": exposure_limit,
                    "references": references,
                }
            )

        self._items = items

        if auto_sync and items:
            db.upsert_item_bank_entries(items)

    @property
    def items(self) -> List[Dict[str, Any]]:
        return list(self._items)

    # ------------------------------------------------------------------
    # selection helpers
    # ------------------------------------------------------------------
    def filter_items(
        self,
        *,
        domain: Optional[str] = None,
        skill_id: Optional[str] = None,
        bloom_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = self._items
        if domain:
            domain_lower = domain.lower()
            results = [item for item in results if item["domain"] == domain_lower]
        if skill_id:
            results = [item for item in results if item["skill_id"] == skill_id]
        if bloom_level:
            results = [item for item in results if item["bloom_level"] == bloom_level]
        return list(results)

    def collect_references(
        self,
        *,
        domain: Optional[str] = None,
        skill_id: Optional[str] = None,
        bloom_level: Optional[str] = None,
        limit: int = 5,
    ) -> List[str]:
        """Aggregate unique reference strings for the provided filters."""

        references: List[str] = []
        for item in self.filter_items(domain=domain, skill_id=skill_id, bloom_level=bloom_level):
            for ref in item.get("references") or []:
                ref_str = str(ref)
                if ref_str and ref_str not in references:
                    references.append(ref_str)
            if len(references) >= limit:
                break
        return references[: max(0, int(limit))]

    def select_by_difficulty(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        target: float,
        k: int = 1,
        ensure_exposure: bool = True,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        working = list(items)
        random.shuffle(working)
        exposure_map: Dict[str, Dict[str, Any]] = {}
        if ensure_exposure:
            exposure_map = db.get_item_exposures([item["id"] for item in working])
            working = self._enforce_exposure_limits(working, exposure_map)
            if not working:
                working = list(items)

        def rank(item: Dict[str, Any]) -> tuple[float, float]:
            difficulty = item.get("difficulty")
            if isinstance(difficulty, (int, float)):
                diff_score = abs(float(difficulty) - target)
            else:
                diff_score = float("inf")

            exposure_ratio = 0.0
            if ensure_exposure and exposure_map:
                info = exposure_map.get(item["id"], {})
                served = float(info.get("served_count", 0) or 0)
                limit = item.get("exposure_limit")
                if limit and limit > 0:
                    exposure_ratio = served / float(limit)
                else:
                    exposure_ratio = served

            return diff_score, exposure_ratio

        ranked = sorted(working, key=rank)
        return ranked[: max(1, int(k))]

    def select_items(
        self,
        *,
        domain: Optional[str] = None,
        skill_id: Optional[str] = None,
        bloom_level: Optional[str] = None,
        target_difficulty: Optional[float] = None,
        k: int = 1,
        exclude: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        candidates = self.filter_items(domain=domain, skill_id=skill_id, bloom_level=bloom_level)
        if exclude:
            exclusion = {ex_id for ex_id in exclude}
            candidates = [item for item in candidates if item["id"] not in exclusion]

        if not candidates:
            return []

        if target_difficulty is None:
            shuffled = list(candidates)
            random.shuffle(shuffled)
            exposure_map = db.get_item_exposures([item["id"] for item in shuffled])
            balanced = self._enforce_exposure_limits(shuffled, exposure_map)
            if not balanced:
                balanced = shuffled
            return balanced[: max(1, int(k))]

        return self.select_by_difficulty(candidates, target=target_difficulty, k=k)

    def mark_served(self, item_ids: Iterable[str]) -> None:
        for item_id in item_ids:
            db.increment_item_exposure(str(item_id))

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _enforce_exposure_limits(
        items: Sequence[Dict[str, Any]],
        exposure_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        limited: List[Dict[str, Any]] = []
        fallback: List[Dict[str, Any]] = []
        for item in items:
            info = exposure_map.get(item["id"], {})
            served = int(info.get("served_count", 0) or 0)
            limit = item.get("exposure_limit")
            if limit is None or limit <= 0:
                fallback.append(item)
                continue
            if served < limit:
                limited.append(item)
        return limited if limited else fallback

    # ------------------------------------------------------------------
    # alternative constructors/helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_items(cls, items: Sequence[Dict[str, Any]]) -> "ItemBank":
        bank = cls.__new__(cls)
        bank.path = Path("<in-memory>")
        bank._items = list(items)
        return bank


_DEFAULT_BANK: Optional[ItemBank] = None


def get_default_bank() -> ItemBank:
    global _DEFAULT_BANK
    if _DEFAULT_BANK is None:
        _DEFAULT_BANK = ItemBank()
    return _DEFAULT_BANK


def load_items(path: str | Path = "items.json") -> List[Dict[str, Any]]:
    """Return validated items from disk and sync them into the database."""
    return ItemBank(path, auto_sync=True).items


def filter_items(
    items: Sequence[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    skill_id: Optional[str] = None,
    bloom_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    bank = ItemBank.from_items(items)
    return bank.filter_items(domain=domain, skill_id=skill_id, bloom_level=bloom_level)


def collect_references(
    *,
    domain: Optional[str] = None,
    skill_id: Optional[str] = None,
    bloom_level: Optional[str] = None,
    limit: int = 5,
) -> List[str]:
    """Shortcut for retrieving references from the default bank."""

    bank = get_default_bank()
    return bank.collect_references(domain=domain, skill_id=skill_id, bloom_level=bloom_level, limit=limit)


def select_by_difficulty(
    items: Sequence[Dict[str, Any]],
    *,
    target: float,
    k: int = 1,
) -> List[Dict[str, Any]]:
    bank = ItemBank.from_items(items)
    return bank.select_by_difficulty(bank.items, target=target, k=k, ensure_exposure=False)


"""Bloom level configuration loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


class BloomLevelConfigError(ValueError):
    """Raised when ``bloom_levels.json`` contains invalid data."""


@dataclass(frozen=True)
class BloomLevel:
    """Immutable representation of a Bloom level definition."""

    id: str
    label: str
    description: str
    min_score: Optional[float]


class BloomLevelRegistry:
    """Load Bloom taxonomy levels from ``bloom_levels.json``."""

    def __init__(self, path: str | Path | None = None) -> None:
        base_path = Path(__file__).resolve().parent
        self.path = Path(path) if path is not None else base_path / "bloom_levels.json"
        self._levels: List[BloomLevel] = []
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Reload Bloom levels from disk and validate the structure."""

        if not self.path.exists():
            raise FileNotFoundError(f"Bloom levels file not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if not isinstance(raw, list):
            raise BloomLevelConfigError("Bloom levels file must contain a JSON list")

        levels: List[BloomLevel] = []
        seen: set[str] = set()
        for idx, entry in enumerate(raw, start=1):
            if not isinstance(entry, dict):
                raise BloomLevelConfigError(f"Entry #{idx} must be a JSON object")

            if "id" not in entry or not str(entry["id"]).strip():
                raise BloomLevelConfigError(f"Entry #{idx} is missing a non-empty 'id'")
            if "label" not in entry or not str(entry["label"]).strip():
                raise BloomLevelConfigError(f"Entry #{idx} is missing a non-empty 'label'")

            level_id = str(entry["id"]).strip()
            if level_id in seen:
                raise BloomLevelConfigError(f"Duplicate Bloom level id detected: {level_id}")
            seen.add(level_id)

            label = str(entry["label"]).strip()
            description = str(entry.get("description", "")).strip()
            min_score_raw = entry.get("min_score")
            min_score: Optional[float]
            if min_score_raw is None or min_score_raw == "":
                min_score = None
            else:
                try:
                    min_score = float(min_score_raw)
                except (TypeError, ValueError) as exc:
                    raise BloomLevelConfigError(
                        f"Entry {level_id} has non-numeric min_score"
                    ) from exc
                if not 0.0 <= min_score <= 1.0:
                    raise BloomLevelConfigError(
                        f"Entry {level_id} min_score must be within [0, 1]"
                    )

            levels.append(BloomLevel(level_id, label, description, min_score))

        if not levels:
            raise BloomLevelConfigError("Bloom levels file may not be empty")

        # Sort by ``min_score`` when all levels provide a threshold; otherwise keep file order.
        if all(level.min_score is not None for level in levels):
            levels.sort(key=lambda lvl: (lvl.min_score, lvl.id))

        self._levels = levels

    # ------------------------------------------------------------------
    @property
    def levels(self) -> List[BloomLevel]:
        """Return a shallow copy of the known Bloom levels."""

        return list(self._levels)

    def sequence(self) -> Sequence[str]:
        """Return the Bloom level identifiers in ascending order."""

        return tuple(level.id for level in self._levels)

    def k_level_sequence(self, length: int = 3) -> Sequence[str]:
        """Return the first ``length`` Bloom levels for K-level progression."""

        if length <= 0:
            raise ValueError("length must be positive")
        seq = self.sequence()
        if len(seq) < length:
            raise BloomLevelConfigError(
                f"Bloom levels file must define at least {length} levels (found {len(seq)})"
            )
        return seq[:length]

    def lowest_level(self) -> str:
        """Return the lowest Bloom level identifier."""

        return self._levels[0].id

    def index(self, level_id: str) -> int:
        """Return the index of ``level_id`` in the ordered sequence."""

        for idx, level in enumerate(self._levels):
            if level.id == level_id:
                return idx
        raise ValueError(f"Unknown Bloom level: {level_id}")

    def get(self, level_id: str) -> Optional[BloomLevel]:
        """Fetch a Bloom level definition if it exists."""

        for level in self._levels:
            if level.id == level_id:
                return level
        return None

    def label_map(self) -> dict[str, str]:
        """Return a mapping from level identifier to label."""

        return {level.id: level.label for level in self._levels}

    def description_map(self) -> dict[str, str]:
        """Return a mapping from level identifier to description."""

        return {level.id: level.description for level in self._levels}

    def thresholds_descending(self) -> List[Tuple[float, str]]:
        """Return ``(min_score, level_id)`` pairs sorted from highest to lowest threshold."""

        pairs: List[Tuple[float, str]] = []
        for level in self._levels:
            if level.min_score is None:
                continue
            pairs.append((level.min_score, level.id))
        pairs.sort(key=lambda item: item[0], reverse=True)
        return pairs

    def formatted_overview(self) -> str:
        """Return a bullet list describing each Bloom level."""

        lines = []
        for level in self._levels:
            summary = f"{level.id}: {level.label}"
            if level.description:
                summary += f" – {level.description}"
            lines.append(f"- {summary}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterable[BloomLevel]:
        return iter(self._levels)


BLOOM_LEVELS = BloomLevelRegistry()
"""Singleton registry used throughout the application."""


"""Utilities for loading master prompt variants used by the tutor."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping

_PROMPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MasterPrompt:
    """Representation of a structured tutor master prompt."""

    id: str
    variant: str
    prompt_version: str
    label: str
    description: str
    system_template: str
    json_instructions: str

    @property
    def normalized_variant(self) -> str:
        return self.variant.lower()


def _load_prompt(path: Path) -> MasterPrompt:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "id",
        "variant",
        "prompt_version",
        "label",
        "description",
        "system_template",
        "json_instructions",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Prompt file {path.name} missing keys: {', '.join(missing)}")
    return MasterPrompt(
        id=str(payload["id"]),
        variant=str(payload["variant"]),
        prompt_version=str(payload["prompt_version"]),
        label=str(payload["label"]),
        description=str(payload["description"]),
        system_template=str(payload["system_template"]),
        json_instructions=str(payload["json_instructions"]),
    )


def _iter_prompt_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.json")):
        if path.is_file():
            yield path


@lru_cache(maxsize=1)
def load_prompts(directory: Path | None = None) -> Mapping[str, MasterPrompt]:
    base_dir = Path(directory) if directory else _PROMPT_DIR
    prompts: Dict[str, MasterPrompt] = {}
    for file_path in _iter_prompt_files(base_dir):
        prompt = _load_prompt(file_path)
        key = prompt.normalized_variant
        if key in prompts:
            raise ValueError(f"Duplicate master prompt variant detected: {prompt.variant}")
        prompts[key] = prompt
    if not prompts:
        raise RuntimeError(f"No master prompt definitions found in {base_dir}")
    return prompts


def get_prompt(variant: str | None) -> MasterPrompt:
    prompts = load_prompts()
    if not variant:
        return next(iter(prompts.values()))
    key = str(variant).lower()
    if key not in prompts:
        raise KeyError(f"Unknown master prompt variant '{variant}'. Available: {', '.join(sorted(prompts))}")
    return prompts[key]


__all__ = ["MasterPrompt", "load_prompts", "get_prompt"]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}

	content = path.read_text(encoding="utf-8").strip()
	if not content:
		return {}

	parsed = yaml.safe_load(content)
	return parsed if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class Thresholds:
	plagiarism_semantic_weight: float = 0.4
	plagiarism_token_weight: float = 0.3
	plagiarism_structure_weight: float = 0.3
	ai_semantic_weight: float = 0.6
	ai_structure_weight: float = 0.4
	high_confidence_min_similarity: float = 0.8
	high_confidence_max_spread: float = 0.2
	high_confidence_min_lines: int = 12
	medium_confidence_min_similarity: float = 0.5
	medium_confidence_min_lines: int = 6
	low_confidence_max_lines: int = 4


@dataclass(frozen=True)
class Settings:
	docs_url: str = "/docs"
	redoc_url: str = "/redoc"
	openapi_url: str = "/openapi.json"


def load_thresholds() -> Thresholds:
	raw = _read_yaml(CONFIG_DIR / "thresholds.yaml")
	return Thresholds(**{k: v for k, v in raw.items() if k in Thresholds.__annotations__})


def load_settings() -> Settings:
	raw = _read_yaml(CONFIG_DIR / "settings.yaml")
	return Settings(**{k: v for k, v in raw.items() if k in Settings.__annotations__})

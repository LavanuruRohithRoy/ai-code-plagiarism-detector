from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.api.file_validation import EXTENSION_LANGUAGE_MAP, normalize_language
from src.pipeline.normalizer import CodeNormalizer
from src.utils.paths import get_raw_data_dir


class DatasetMatcher:
    def __init__(self, root: str | Path | None = None, normalizer: CodeNormalizer | None = None):
        self.root = Path(root) if root is not None else get_raw_data_dir()
        self.normalizer = normalizer or CodeNormalizer()
        self._index: dict[str, dict[str, str]] = {}
        self._entries: list[dict[str, str | set[str]]] = []
        self._build_index()

    def _normalized_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _iter_corpus_files(self):
        sources = [
            ("AI", self.root / "ai"),
            ("HUMAN", self.root / "human"),
        ]

        for label, source_root in sources:
            if not source_root.exists():
                continue

            for path in source_root.rglob("*"):
                if not path.is_file():
                    continue

                ext = path.suffix.lower()
                language = EXTENSION_LANGUAGE_MAP.get(ext)
                normalized_language, language_error = normalize_language(language)
                if language_error or not normalized_language:
                    continue

                yield label, normalized_language, path

    def _tokenize(self, code: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", code.lower()))

    def _build_index(self):
        for label, language, path in self._iter_corpus_files():
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not raw or not raw.strip():
                continue

            normalized_code, _ = self.normalizer.normalize(raw)
            code_hash = self._normalized_hash(normalized_code)

            if code_hash not in self._index:
                self._index[code_hash] = {
                    "label": label,
                    "language": language,
                    "path": str(path).replace("\\", "/"),
                }

            self._entries.append(
                {
                    "label": label,
                    "language": language,
                    "path": str(path).replace("\\", "/"),
                    "filename": path.name,
                    "normalized_code": normalized_code,
                    "tokens": self._tokenize(normalized_code),
                }
            )

    def size(self) -> int:
        return len(self._index)

    def find_by_normalized_code(self, normalized_code: str) -> dict[str, str] | None:
        code_hash = self._normalized_hash(normalized_code)
        return self._index.get(code_hash)

    def find_best_pattern_match(
        self,
        normalized_code: str,
        language: str | None = None,
        min_score: float = 0.12,
    ) -> dict[str, str] | None:
        query_tokens = self._tokenize(normalized_code)
        if not query_tokens:
            return None

        best: dict[str, str] | None = None
        best_score = 0.0

        for entry in self._entries:
            entry_language = str(entry["language"])
            if language and entry_language != language:
                continue

            candidate_tokens = entry["tokens"]
            if not isinstance(candidate_tokens, set) or not candidate_tokens:
                continue

            intersection = len(query_tokens & candidate_tokens)
            union = len(query_tokens | candidate_tokens)
            if union == 0:
                continue

            score = intersection / union
            if score > best_score:
                best_score = score
                best = {
                    "label": str(entry["label"]),
                    "language": entry_language,
                    "path": str(entry["path"]),
                    "filename": str(entry["filename"]),
                    "match_score": f"{score:.3f}",
                    "match_type": "pattern",
                    "normalized_code": str(entry["normalized_code"]),
                }

        if best is None or best_score < min_score:
            return None
        return best

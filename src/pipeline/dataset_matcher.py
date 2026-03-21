from __future__ import annotations

import hashlib
from pathlib import Path

from src.api.file_validation import EXTENSION_LANGUAGE_MAP, normalize_language
from src.pipeline.normalizer import CodeNormalizer
from src.utils.paths import get_raw_data_dir


class DatasetMatcher:
    def __init__(self, root: str | Path | None = None, normalizer: CodeNormalizer | None = None):
        self.root = Path(root) if root is not None else get_raw_data_dir()
        self.normalizer = normalizer or CodeNormalizer()
        self._index: dict[str, dict[str, str]] = {}
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

    def size(self) -> int:
        return len(self._index)

    def find_by_normalized_code(self, normalized_code: str) -> dict[str, str] | None:
        code_hash = self._normalized_hash(normalized_code)
        return self._index.get(code_hash)

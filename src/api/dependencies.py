from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.orchestrator import AnalysisPipeline
from src.storage.faiss_index import faiss_index
from src.storage.repository import AnalysisRepository
from src.utils.paths import get_embeddings_dir


_repository = AnalysisRepository()
_pipeline = AnalysisPipeline()

_FAISS_CACHE_DIR = get_embeddings_dir()
_FAISS_INDEX_FILE = _FAISS_CACHE_DIR / "faiss.index"
_FAISS_META_FILE = _FAISS_CACHE_DIR / "faiss.meta.json"


def get_repository() -> AnalysisRepository:
	return _repository


def get_faiss_index():
	return faiss_index


def get_pipeline() -> AnalysisPipeline:
	return _pipeline


def _read_cached_embedding_count() -> int | None:
	if not _FAISS_META_FILE.exists():
		return None

	try:
		payload = json.loads(_FAISS_META_FILE.read_text(encoding="utf-8"))
		value = payload.get("embedding_count")
		return int(value) if value is not None else None
	except Exception:
		return None


def _write_cached_embedding_count(embedding_count: int) -> None:
	_FAISS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	_FAISS_META_FILE.write_text(
		json.dumps({"embedding_count": int(embedding_count)}, indent=2),
		encoding="utf-8",
	)


def sync_faiss_with_db() -> int:
	embedding_count = _repository.count_embeddings()

	if embedding_count == 0:
		faiss_index.clear()
		return 0

	cached_count = _read_cached_embedding_count()
	can_use_cache = (
		cached_count == embedding_count
		and _FAISS_INDEX_FILE.exists()
	)

	if can_use_cache and faiss_index.load(str(_FAISS_INDEX_FILE)):
		return faiss_index.size()

	embeddings = _repository.fetch_all_embeddings(expected_dim=faiss_index.vector_dim)
	faiss_index.rebuild(embeddings)
	faiss_index.save(str(_FAISS_INDEX_FILE))
	_write_cached_embedding_count(embedding_count)
	return faiss_index.size()

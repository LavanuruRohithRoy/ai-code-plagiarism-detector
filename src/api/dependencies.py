from __future__ import annotations

from src.pipeline.orchestrator import AnalysisPipeline
from src.storage.faiss_index import faiss_index
from src.storage.repository import AnalysisRepository


_repository = AnalysisRepository()
_pipeline = AnalysisPipeline()


def get_repository() -> AnalysisRepository:
	return _repository


def get_faiss_index():
	return faiss_index


def get_pipeline() -> AnalysisPipeline:
	return _pipeline


def sync_faiss_with_db() -> int:
	embeddings = _repository.fetch_all_embeddings(expected_dim=faiss_index.vector_dim)

	# Always rebuild to avoid stale index state in long-lived dev sessions.
	faiss_index.rebuild(embeddings)
	return faiss_index.size()

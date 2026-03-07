import numpy as np

from src.pipeline.orchestrator import AnalysisPipeline


class _DummyEmbedding:
	def __init__(self, vec):
		self._vec = vec

	def detach(self):
		return self

	def numpy(self):
		return self._vec


class _DummyNormalizer:
	def normalize(self, code):
		return code, {}


class _DummyAst:
	def analyze(self, _code):
		return {"num_functions": 1}


class _DummyToken:
	def jaccard_similarity(self, _a, _b):
		return 0.75


class _DummyEmbedder:
	def generate(self, _code):
		return _DummyEmbedding(np.array([0.2, 0.1, 0.4], dtype=np.float32))


class _DummyScorer:
	def compute_plagiarism_score(self, *_args):
		return 74.0

	def compute_ai_probability(self, *_args):
		return 66.0

	def compute_confidence(self, **_kwargs):
		return "medium"


class _DummySearch:
	def top_similarity(self, _vector):
		return 0.9


class _DummyFaiss:
	def __init__(self):
		self.add_calls = 0

	def add(self, _vector):
		self.add_calls += 1


class _DummyRepo:
	def __init__(self, existing=True, insert_ok=True):
		self.existing = existing
		self.insert_ok = insert_ok

	def fetch_all_for_similarity(self):
		if not self.existing:
			return []
		return [("print('x')", "{\"num_functions\": 1}")]

	def save_result(self, *_args, **_kwargs):
		return self.insert_ok


def _build_pipeline(repo):
	pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
	pipeline.normalizer = _DummyNormalizer()
	pipeline.ast_analyzer = _DummyAst()
	pipeline.token_similarity = _DummyToken()
	pipeline.embedding_generator = _DummyEmbedder()
	pipeline.scorer = _DummyScorer()
	pipeline.faiss_search = _DummySearch()
	pipeline.faiss_index = _DummyFaiss()
	pipeline.repo = repo
	return pipeline


def test_duplicate_insert_does_not_add_faiss_vector():
	pipeline = _build_pipeline(_DummyRepo(existing=True, insert_ok=False))

	result = pipeline.run("print('hello')", language="python")

	assert pipeline.faiss_index.add_calls == 0
	assert result["explanation"]["db_inserted"] is False


def test_successful_insert_adds_faiss_vector_once():
	pipeline = _build_pipeline(_DummyRepo(existing=True, insert_ok=True))

	result = pipeline.run("print('hello world')", language="python")

	assert pipeline.faiss_index.add_calls == 1
	assert result["explanation"]["db_inserted"] is True


def test_first_submission_returns_baseline_low_confidence():
	pipeline = _build_pipeline(_DummyRepo(existing=False, insert_ok=True))

	result = pipeline.run("print('first')", language="python")

	assert result["plagiarism_percentage"] == 0.0
	assert result["confidence"] == "low"
	assert pipeline.faiss_index.add_calls == 1

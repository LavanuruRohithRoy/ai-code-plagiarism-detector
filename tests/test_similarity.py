from src.pipeline.scorer import ScoreAggregator
from src.utils.config import Thresholds


def test_confidence_high_when_signals_agree_and_long_code():
	scorer = ScoreAggregator(
		Thresholds(
			high_confidence_min_similarity=0.8,
			high_confidence_max_spread=0.2,
			high_confidence_min_lines=12,
		)
	)

	confidence = scorer.compute_confidence(
		token_similarity=0.82,
		semantic_similarity=0.85,
		structure_similarity=0.8,
		line_count=20,
	)

	assert confidence == "high"


def test_confidence_low_for_tiny_samples():
	scorer = ScoreAggregator()

	confidence = scorer.compute_confidence(
		token_similarity=0.95,
		semantic_similarity=0.93,
		structure_similarity=0.9,
		line_count=3,
	)

	assert confidence == "low"


def test_scores_are_bounded_to_valid_percentage_range():
	scorer = ScoreAggregator()

	plagiarism = scorer.compute_plagiarism_score(10.0, 10.0, 10.0)
	ai_prob = scorer.compute_ai_probability(10.0, 10.0)

	assert plagiarism == 100.0
	assert ai_prob == 100.0

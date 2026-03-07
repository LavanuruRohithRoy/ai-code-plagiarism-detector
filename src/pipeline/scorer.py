from src.utils.config import Thresholds, load_thresholds


class ScoreAggregator:
    """
    Combines multiple similarity signals into final scores.
    """

    def __init__(self, thresholds: Thresholds | None = None):
        self.thresholds = thresholds or load_thresholds()

    def compute_plagiarism_score(
        self,
        token_similarity: float,
        semantic_similarity: float,
        structure_similarity: float
    ) -> float:
        score = (
            self.thresholds.plagiarism_semantic_weight * semantic_similarity +
            self.thresholds.plagiarism_token_weight * token_similarity +
            self.thresholds.plagiarism_structure_weight * structure_similarity
        )
        return round(max(0.0, min(100.0, score * 100)), 2)

    def compute_ai_probability(
        self,
        semantic_similarity: float,
        structure_similarity: float
    ) -> float:
        ai_score = (
            self.thresholds.ai_semantic_weight * semantic_similarity +
            self.thresholds.ai_structure_weight * structure_similarity
        )
        return round(max(0.0, min(100.0, ai_score * 100)), 2)

    def compute_confidence(
        self,
        token_similarity: float,
        semantic_similarity: float,
        structure_similarity: float,
        line_count: int,
    ) -> str:
        if line_count <= self.thresholds.low_confidence_max_lines:
            return "low"

        agreement = max(token_similarity, semantic_similarity, structure_similarity)
        spread = max(token_similarity, semantic_similarity, structure_similarity) - min(
            token_similarity,
            semantic_similarity,
            structure_similarity,
        )

        if (
            agreement >= self.thresholds.high_confidence_min_similarity
            and spread <= self.thresholds.high_confidence_max_spread
            and line_count >= self.thresholds.high_confidence_min_lines
        ):
            return "high"

        if (
            agreement >= self.thresholds.medium_confidence_min_similarity
            and line_count >= self.thresholds.medium_confidence_min_lines
        ):
            return "medium"

        return "low"

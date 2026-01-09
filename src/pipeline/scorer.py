class ScoreAggregator:
    """
    Combines multiple similarity signals into final scores.
    """

    def compute_plagiarism_score(
        self,
        token_similarity: float,
        semantic_similarity: float,
        structure_similarity: float
    ) -> float:
        score = (
            0.4 * semantic_similarity +
            0.3 * token_similarity +
            0.3 * structure_similarity
        )
        return round(score * 100, 2)

    def compute_ai_probability(
        self,
        semantic_similarity: float,
        structure_similarity: float
    ) -> float:
        ai_score = (
            0.6 * semantic_similarity +
            0.4 * structure_similarity
        )
        return round(ai_score * 100, 2)

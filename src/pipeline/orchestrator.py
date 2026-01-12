# src/pipeline/orchestrator.py

import hashlib
import json

from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.token_similarity import TokenSimilarity
from src.pipeline.embedding import EmbeddingGenerator
from src.pipeline.scorer import ScoreAggregator
from src.storage.faiss_index import FaissIndex
from src.storage.repository import AnalysisRepository


class AnalysisPipeline:
    """
    Central orchestration layer.
    Stable, defensive, and state-safe.
    """

    def __init__(self):
        self.normalizer = CodeNormalizer()
        self.ast_analyzer = ASTAnalyzer()
        self.token_similarity = TokenSimilarity()
        self.embedding_generator = EmbeddingGenerator()
        self.scorer = ScoreAggregator()
        self.repo = AnalysisRepository()

        # CodeBERT embedding dimension
        self.faiss_index = FaissIndex(vector_dim=768)

    def _ast_similarity(self, a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        keys = set(a.keys()) | set(b.keys())
        diff = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
        return 1 / (1 + diff)

    def run(self, code: str, language: str | None = None) -> dict:

        # 1️⃣ Normalize
        normalized_code, _ = self.normalizer.normalize(code)

        # 2️⃣ AST
        ast_features = {}
        if language in (None, "python"):
            ast_features = self.ast_analyzer.analyze(normalized_code)

        # 3️⃣ Embedding
        embedding = self.embedding_generator.generate(normalized_code)
        vector = embedding.detach().numpy()

        # 4️⃣ FIRST SUBMISSION (HARD BASELINE)
        if self.faiss_index.size() == 0:
            self.faiss_index.add(vector)

            code_hash = hashlib.sha256(code.encode()).hexdigest()
            self.repo.save_result(
                code_hash=code_hash,
                plagiarism_score=0.0,
                ai_probability=0.0,
                normalized_code=normalized_code,
                ast_features=ast_features
            )

            return {
                "plagiarism_percentage": 0.0,
                "ai_probability": 0.0,
                "confidence": "low"
            }

        # 5️⃣ SEMANTIC SIMILARITY (SEARCH BEFORE ADD)
        distances, _ = self.faiss_index.search(vector, k=1)
        semantic_sim = 0.0
        if distances and len(distances) > 0:
            semantic_sim = 1 / (1 + float(distances[0]))

        # 6️⃣ TOKEN + AST SIMILARITY (REAL HISTORY)
        stored_rows = self.repo.fetch_all_for_similarity()

        token_sim = 0.0
        structure_sim = 0.0

        for norm_db, ast_json in stored_rows:
            # Token similarity
            if norm_db:
                ts = self.token_similarity.jaccard_similarity(
                    normalized_code, norm_db
                )
                token_sim = max(token_sim, ts)

            # AST similarity
            if ast_json:
                try:
                    ast_old = json.loads(ast_json)
                    ss = self._ast_similarity(ast_features, ast_old)
                    structure_sim = max(structure_sim, ss)
                except Exception:
                    continue

        # 7️⃣ SCORE AGGREGATION
        plagiarism_score = self.scorer.compute_plagiarism_score(
            token_similarity=token_sim,
            semantic_similarity=semantic_sim,
            structure_similarity=structure_sim
        )

        ai_probability = self.scorer.compute_ai_probability(
            semantic_similarity=semantic_sim,
            structure_similarity=structure_sim
        )

        # 8️⃣ SIZE-BASED CALIBRATION (CRITICAL)
        line_count = len([l for l in normalized_code.splitlines() if l.strip()])

        if line_count <= 4:
            plagiarism_score = min(plagiarism_score, 35.0)
        elif line_count <= 8:
            plagiarism_score = min(plagiarism_score, 60.0)

        # 9️⃣ STORE AFTER COMPARISON
        self.faiss_index.add(vector)

        code_hash = hashlib.sha256(code.encode()).hexdigest()
        self.repo.save_result(
            code_hash=code_hash,
            plagiarism_score=plagiarism_score,
            ai_probability=ai_probability,
            normalized_code=normalized_code,
            ast_features=ast_features
        )

        return {
            "plagiarism_percentage": round(plagiarism_score, 2),
            "ai_probability": round(ai_probability, 2),
            "confidence": "medium"
        }

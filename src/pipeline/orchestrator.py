import hashlib
import json
import re

from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.token_similarity import TokenSimilarity
from src.pipeline.embedding import EmbeddingGenerator
from src.pipeline.faiss_search import FaissSearch
from src.pipeline.scorer import ScoreAggregator

from src.storage.repository import AnalysisRepository
from src.storage.faiss_index import faiss_index


class AnalysisPipeline:

    def __init__(self):
        self.normalizer = CodeNormalizer()
        self.ast_analyzer = ASTAnalyzer()
        self.token_similarity = TokenSimilarity()
        self.embedding_generator = EmbeddingGenerator()
        self.scorer = ScoreAggregator()

        self.repo = AnalysisRepository()

        # shared FAISS instance
        self.faiss_index = faiss_index
        self.faiss_search = FaissSearch(self.faiss_index)

    def _ast_similarity(self, a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0

        keys = set(a.keys()) | set(b.keys())
        diff = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)

        return 1 / (1 + diff)

    def _heuristic_structure_features(self, code: str) -> dict[str, int]:
        # Language-agnostic approximation for non-Python code paths.
        non_empty = [line.strip() for line in code.splitlines() if line.strip()]
        loop_count = len(re.findall(r"\b(for|while|do)\b", code))
        cond_count = len(re.findall(r"\b(if|else if|switch|case)\b", code))
        fn_like = len(
            re.findall(
                r"\b[A-Za-z_][A-Za-z0-9_<>:\[\]]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*\{",
                code,
            )
        )

        depth = 0
        max_depth = 0
        for ch in code:
            if ch == "{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == "}":
                depth = max(0, depth - 1)

        if max_depth == 0:
            indent_depth = 0
            for line in non_empty:
                leading_spaces = len(line) - len(line.lstrip(" "))
                indent_depth = max(indent_depth, leading_spaces // 4)
            max_depth = indent_depth

        return {
            "num_functions": fn_like,
            "num_loops": loop_count,
            "num_conditionals": cond_count,
            "max_nesting_depth": max_depth,
        }

    def _band(self, value: float) -> str:
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"

    def _build_highlights(self, normalized_code: str, size_penalty: bool) -> list[dict[str, str | int]]:
        highlights: list[dict[str, str | int]] = []
        lines = normalized_code.splitlines()

        if size_penalty:
            highlights.append(
                {
                    "kind": "warning",
                    "line_start": 1,
                    "line_end": max(1, len(lines)),
                    "message": "Small snippet: plagiarism score is capped for fairness.",
                }
            )

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            if len(stripped) > 120:
                highlights.append(
                    {
                        "kind": "info",
                        "line_start": i,
                        "line_end": i,
                        "message": "Very long line may indicate compressed or generated style.",
                    }
                )

            if stripped.startswith(("def ", "class ", "for ", "while ", "if ")):
                highlights.append(
                    {
                        "kind": "signal",
                        "line_start": i,
                        "line_end": i,
                        "message": "Control-flow/structure line contributes to AST similarity.",
                    }
                )

            if len(highlights) >= 8:
                break

        return highlights

    def run(
        self,
        code: str,
        language: str | None = None,
        input_metrics: dict[str, int] | None = None,
    ) -> dict:

        # -------------------------
        # Normalize code
        # -------------------------
        normalized_code, _ = self.normalizer.normalize(code)

        # -------------------------
        # AST feature extraction
        # -------------------------
        ast_features = {}
        normalized_language = language.lower() if language else None

        if normalized_language in (None, "python"):
            ast_features = self.ast_analyzer.analyze(normalized_code)

            # Fallback to heuristic features when parse-safe AST is empty.
            if not any(ast_features.values()):
                ast_features = self._heuristic_structure_features(normalized_code)
        else:
            ast_features = self._heuristic_structure_features(normalized_code)

        # -------------------------
        # Generate embedding
        # -------------------------
        embedding = self.embedding_generator.generate(normalized_code)
        vector = embedding.detach().numpy()
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        # -------------------------
        # BASELINE CHECK
        # Use DB instead of FAISS
        # -------------------------
        existing_records = self.repo.fetch_all_for_similarity()

        if not existing_records:

            inserted = self.repo.save_result(
                code_hash,
                0.0,
                0.0,
                normalized_code,
                ast_features,
                vector
            )

            if inserted:
                self.faiss_index.add(vector)

            return {
                "plagiarism_percentage": 0.0,
                "ai_probability": 0.0,
                "confidence": "low",
                "explanation": {
                    "reasoning": "First submission baseline"
                }
            }

        # -------------------------
        # Semantic similarity
        # -------------------------
        semantic_sim = self.faiss_search.top_similarity(vector)

        # -------------------------
        # Token + AST similarity
        # -------------------------
        token_sim = 0.0
        structure_sim = 0.0

        for norm_db, ast_json in existing_records:

            if norm_db:
                token_sim = max(
                    token_sim,
                    self.token_similarity.jaccard_similarity(
                        normalized_code,
                        norm_db
                    )
                )

            if ast_json:
                structure_sim = max(
                    structure_sim,
                    self._ast_similarity(
                        ast_features,
                        json.loads(ast_json)
                    )
                )

        # -------------------------
        # Score calculation
        # -------------------------
        plagiarism_score = self.scorer.compute_plagiarism_score(
            token_sim,
            semantic_sim,
            structure_sim
        )

        ai_probability = self.scorer.compute_ai_probability(
            semantic_sim,
            structure_sim
        )

        # -------------------------
        # Small code penalty
        # -------------------------
        line_count = len([l for l in normalized_code.splitlines() if l.strip()])

        size_penalty = False

        if line_count <= 4:
            plagiarism_score = min(plagiarism_score, 35.0)
            size_penalty = True

        elif line_count <= 8:
            plagiarism_score = min(plagiarism_score, 60.0)
            size_penalty = True

        # -------------------------
        # Update FAISS + DB
        # -------------------------
        inserted = self.repo.save_result(
            code_hash,
            plagiarism_score,
            ai_probability,
            normalized_code,
            ast_features,
            vector
        )

        if inserted:
            self.faiss_index.add(vector)

        # -------------------------
        # Reasoning text
        # -------------------------
        reasoning = (
            "Small code sample; score capped to avoid overestimation."
            if size_penalty
            else "Similarity based on semantic, token, and structural overlap."
        )

        if normalized_language and normalized_language != "python":
            reasoning += " Structure signal uses language-agnostic heuristics for non-Python code."

        signal_bands = {
            "token": self._band(token_sim),
            "semantic": self._band(semantic_sim),
            "structure": self._band(structure_sim),
        }

        highlights = self._build_highlights(normalized_code, size_penalty)
        metrics = input_metrics or {
            "total_lines": len(normalized_code.splitlines()),
            "non_empty_lines": line_count,
            "char_count": len(code),
            "token_count_estimate": len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", normalized_code)),
            "comment_lines_estimate": 0,
        }

        # -------------------------
        # Final response
        # -------------------------
        return {
            "plagiarism_percentage": round(plagiarism_score, 2),
            "ai_probability": round(ai_probability, 2),
            "confidence": self.scorer.compute_confidence(
                token_similarity=token_sim,
                semantic_similarity=semantic_sim,
                structure_similarity=structure_sim,
                line_count=line_count
            ),
            "explanation": {
                "token_similarity": round(token_sim, 3),
                "semantic_similarity": round(semantic_sim, 3),
                "structure_similarity": round(structure_sim, 3),
                "code_lines": line_count,
                "size_penalty_applied": size_penalty,
                "db_inserted": inserted,
                "language": normalized_language,
                "metrics": metrics,
                "signal_bands": signal_bands,
                "highlights": highlights,
                "reasoning": reasoning
            }
        }
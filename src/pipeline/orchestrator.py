import hashlib
import json
import re

from src.pipeline.normalizer import CodeNormalizer
from src.pipeline.ast_analyzer import ASTAnalyzer
from src.pipeline.token_similarity import TokenSimilarity
from src.pipeline.embedding import EmbeddingGenerator
from src.pipeline.faiss_search import FaissSearch
from src.pipeline.scorer import ScoreAggregator
from src.pipeline.structure_features import extract_structure_features, heuristic_structure_features
from src.pipeline.dataset_matcher import DatasetMatcher

from src.storage.repository import AnalysisRepository
from src.storage.faiss_index import faiss_index
from src.utils.paths import get_embeddings_dir


class AnalysisPipeline:

    def __init__(self):
        self.normalizer = CodeNormalizer()
        self.ast_analyzer = ASTAnalyzer()
        self.token_similarity = TokenSimilarity()
        self.embedding_generator = EmbeddingGenerator()
        self.scorer = ScoreAggregator()
        self.dataset_matcher = DatasetMatcher(normalizer=self.normalizer)

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
        return heuristic_structure_features(code)

    def _band(self, value: float) -> str:
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"

    def _line_char_offsets(self, code: str) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for line in code.splitlines(keepends=True):
            line_start = cursor
            line_end = cursor + len(line.rstrip("\n\r"))
            offsets.append((line_start, max(line_start, line_end)))
            cursor += len(line)
        if not offsets and code:
            offsets.append((0, len(code)))
        return offsets

    def _build_highlights(
        self,
        normalized_code: str,
        size_penalty: bool,
        token_sim: float,
        semantic_sim: float,
        structure_sim: float,
    ) -> list[dict[str, str | int]]:
        highlights: list[dict[str, str | int]] = []
        lines = normalized_code.splitlines()
        offsets = self._line_char_offsets(normalized_code)

        if size_penalty and normalized_code:
            highlights.append(
                {
                    "start": 0,
                    "end": len(normalized_code),
                    "type": "plagiarism",
                }
            )

        for index, line in enumerate(lines):
            i = index + 1
            stripped = line.strip()
            if not stripped:
                continue

            if index < len(offsets):
                line_start, line_end = offsets[index]
            else:
                line_start, line_end = (0, 0)

            if line_end <= line_start:
                continue

            if len(stripped) > 120:
                highlights.append(
                    {
                        "start": line_start,
                        "end": line_end,
                        "type": "ai-detected" if semantic_sim >= token_sim else "plagiarism",
                    }
                )

            if stripped.startswith(("def ", "class ", "for ", "while ", "if ")):
                highlights.append(
                    {
                        "start": line_start,
                        "end": line_end,
                        "type": "plagiarism" if structure_sim >= 0.2 else "ai-detected",
                    }
                )

            if len(highlights) >= 8:
                break

        return highlights

    def _highlight_legend(self) -> dict[str, str]:
        return {
            "plagiarism": "Red highlights indicate segments strongly associated with overlap/similarity to known corpus patterns.",
            "ai-detected": "Yellow highlights indicate segments with AI-like stylistic or structural signals.",
        }

    def _persist_faiss_cache(self) -> None:
        try:
            cache_dir = get_embeddings_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)

            index_file = cache_dir / "faiss.index"
            meta_file = cache_dir / "faiss.meta.json"

            self.faiss_index.save(str(index_file))
            embedding_count = self.repo.count_embeddings()
            meta_file.write_text(
                json.dumps({"embedding_count": int(embedding_count)}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _build_exact_match_response(
        self,
        normalized_code: str,
        normalized_language: str | None,
        input_metrics: dict[str, int] | None,
        known_match: dict[str, str],
    ) -> dict:
        label = known_match.get("label", "UNKNOWN")
        line_count = len([line for line in normalized_code.splitlines() if line.strip()])
        metrics = input_metrics or {
            "total_lines": len(normalized_code.splitlines()),
            "non_empty_lines": line_count,
            "char_count": len(normalized_code),
            "token_count_estimate": len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", normalized_code)),
            "comment_lines_estimate": 0,
        }

        ai_probability = 8.0 if label == "HUMAN" else 92.0 if label == "AI" else 50.0

        return {
            "plagiarism_percentage": 99.0,
            "ai_probability": ai_probability,
            "confidence": "high",
            "explanation": {
                "token_similarity": 1.0,
                "semantic_similarity": 1.0,
                "structure_similarity": 1.0,
                "code_lines": line_count,
                "size_penalty_applied": False,
                "db_inserted": False,
                "language": normalized_language,
                "metrics": metrics,
                "signal_bands": {
                    "token": "high",
                    "semantic": "high",
                    "structure": "high",
                },
                "highlights": [
                    {
                        "start": 0,
                        "end": len(normalized_code),
                        "type": "plagiarism",
                    }
                ] if normalized_code else [],
                "source_code": normalized_code,
                "highlight_legend": self._highlight_legend(),
                "known_match": known_match,
                "reasoning": (
                    f"Exact normalized match found in dataset ({known_match.get('path', 'unknown')}). "
                    f"Source label: {label}."
                ),
            },
        }

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
        known_match = self.dataset_matcher.find_by_normalized_code(normalized_code)

        # -------------------------
        # AST feature extraction
        # -------------------------
        ast_features = extract_structure_features(
            code=normalized_code,
            language=language,
            ast_analyzer=self.ast_analyzer,
        )
        normalized_language = language.lower() if language else None

        if known_match:
            return self._build_exact_match_response(
                normalized_code=normalized_code,
                normalized_language=normalized_language,
                input_metrics=input_metrics,
                known_match=known_match,
            )

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
                    "source_code": normalized_code,
                    "highlight_legend": self._highlight_legend(),
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

        # Conservative damping to reduce false positives on human-written code.
        agreement = max(token_sim, semantic_sim, structure_sim)
        if agreement < 0.35:
            plagiarism_score *= 0.45
            ai_probability *= 0.4
        elif token_sim < 0.15 and structure_sim < 0.25 and semantic_sim < 0.45:
            ai_probability *= 0.6

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

        if line_count <= 10:
            ai_probability = min(ai_probability, 55.0)

        plagiarism_score = max(0.0, min(plagiarism_score, 100.0))
        ai_probability = max(0.0, min(ai_probability, 100.0))

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
            self._persist_faiss_cache()

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

        highlights = self._build_highlights(
            normalized_code,
            size_penalty,
            token_sim,
            semantic_sim,
            structure_sim,
        )
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
                "source_code": normalized_code,
                "highlight_legend": self._highlight_legend(),
                "reasoning": reasoning
            }
        }
from src.storage.db import SessionLocal
from src.storage.models import AnalysisResult
from sqlalchemy.exc import IntegrityError
import json
import numpy as np


class AnalysisRepository:

    def save_result(
        self,
        code_hash,
        plagiarism_score,
        ai_probability,
        normalized_code,
        ast_features,
        embedding_vector
    ) -> bool:
        db = SessionLocal()

        try:
            embedding_bytes = np.array(embedding_vector).astype("float32").tobytes()

            result = AnalysisResult(
                code_hash=code_hash,
                plagiarism_score=plagiarism_score,
                ai_probability=ai_probability,
                normalized_code=normalized_code,
                ast_features=json.dumps(ast_features),
                embedding=embedding_bytes
            )

            db.add(result)
            db.commit()
            return True

        except IntegrityError:
            db.rollback()
            return False

        finally:
            db.close()

    def fetch_all_for_similarity(self):
        db = SessionLocal()

        try:
            return db.query(
                AnalysisResult.normalized_code,
                AnalysisResult.ast_features
            ).all()

        finally:
            db.close()

    def exists_code_hash(self, code_hash: str) -> bool:
        db = SessionLocal()

        try:
            return (
                db.query(AnalysisResult.id)
                .filter(AnalysisResult.code_hash == code_hash)
                .first()
                is not None
            )

        finally:
            db.close()

    def count_results(self) -> int:
        db = SessionLocal()

        try:
            return db.query(AnalysisResult).count()

        finally:
            db.close()

    # NEW: used to rebuild FAISS index
    def fetch_all_embeddings(self, expected_dim: int = 768):
        db = SessionLocal()

        try:
            results = db.query(AnalysisResult.embedding).all()

            embeddings = []

            for row in results:
                if row.embedding is None:
                    continue

                vector = np.frombuffer(row.embedding, dtype=np.float32)

                # ensure correct dimension
                if vector.shape[0] == expected_dim:
                    embeddings.append(vector)

            return embeddings

        finally:
            db.close()
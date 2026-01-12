from src.storage.db import SessionLocal
from src.storage.models import AnalysisResult
import json
class AnalysisRepository:

    def save_result(
        self,
        code_hash,
        plagiarism_score,
        ai_probability,
        normalized_code,
        ast_features
    ):
        db = SessionLocal()
        try:
            result = AnalysisResult(
                code_hash=code_hash,
                plagiarism_score=plagiarism_score,
                ai_probability=ai_probability,
                normalized_code=normalized_code,
                ast_features=json.dumps(ast_features)
            )
            db.add(result)
            db.commit()
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

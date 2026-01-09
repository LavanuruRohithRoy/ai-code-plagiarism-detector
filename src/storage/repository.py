from src.storage.db import SessionLocal
from src.storage.models import AnalysisResult


class AnalysisRepository:
    def save_result(
        self,
        code_hash: str,
        plagiarism_score: float,
        ai_probability: float
    ):
        db = SessionLocal()
        try:
            result = AnalysisResult(
                code_hash=code_hash,
                plagiarism_score=plagiarism_score,
                ai_probability=ai_probability
            )
            db.add(result)
            db.commit()
        finally:
            db.close()

from sqlalchemy import Column, Integer, Float, String, DateTime, Text
from datetime import datetime
from src.storage.db import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    code_hash = Column(String, unique=True, index=True)
    plagiarism_score = Column(Float)
    ai_probability = Column(Float)
    normalized_code = Column(Text)
    ast_features = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

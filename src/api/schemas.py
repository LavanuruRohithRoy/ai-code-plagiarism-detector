from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=50000)
    language: Optional[str] = None


class AnalyzeExplanation(BaseModel):
    token_similarity: Optional[float] = None
    semantic_similarity: Optional[float] = None
    structure_similarity: Optional[float] = None
    code_lines: Optional[int] = None
    size_penalty_applied: Optional[bool] = None
    db_inserted: Optional[bool] = None
    language: Optional[str] = None
    metrics: Optional[dict[str, int]] = None
    signal_bands: Optional[dict[str, str]] = None
    highlights: Optional[list[dict[str, str | int]]] = None
    reasoning: str


class AnalyzeResponse(BaseModel):
    plagiarism_percentage: float
    ai_probability: float
    confidence: str
    explanation: AnalyzeExplanation


class FileAnalyzeResponse(BaseModel):
    filename: str
    language: Optional[str] = None
    plagiarism_percentage: float
    ai_probability: float
    confidence: str
    explanation: AnalyzeExplanation


class BatchAnalyzeResponse(BaseModel):
    total_files: int
    succeeded: int
    failed: int
    results: list[FileAnalyzeResponse]
    errors: dict[str, str] = Field(default_factory=dict)
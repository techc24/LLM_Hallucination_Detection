from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Domain(str, Enum):
    finance = "finance"
    law = "law"
    healthcare = "healthcare"


class AnalyzeRequest(BaseModel):
    domain: Domain
    question: str
    answer: Optional[str] = None
    use_mock_llm: bool = Field(default=False, description="Use mock LLM instead of OpenAI")


class ClaimResult(BaseModel):
    claim: str
    domain: Domain
    evidence_snippets: List[str]
    label: str
    evidence_score: float
    confidence_flags: List[str] = Field(default_factory=list)
    nli_relation: str = "neutral"


class AnalyzeResponse(BaseModel):
    domain: Domain
    question: str
    answer: str
    claims: List[ClaimResult]
    hallucination_score: float
    explanation: str
    score_breakdown: Dict[str, float]
    analysis_details: Dict[str, object]


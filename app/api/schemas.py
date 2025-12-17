# filepath: app/api/schemas.py
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LawSourceItem(BaseModel):
    id: str
    score: float
    dataset: Optional[str] = None
    category: Optional[str] = None
    question: Optional[str] = None
    answerSnippet: Optional[str] = None
    docNo: Optional[int] = None


class LawChatRequest(BaseModel):
    """
    클라이언트 JSON 예시:
    {
        "question": "집주인 전세보증금 안줄 때 경매 진행 방법",
        "topK": 5
    }
    """
    question: str = Field(..., min_length=1, max_length=500)

    # JSON 키: topK  →  Python 내부 필드명: top_k
    top_k: int = Field(
        5,
        alias="topK",
        ge=1,
        le=10,
        description="검색할 상위 문서 개수",
    )


class LawChatResult(BaseModel):
    answer: str
    sources: List[LawSourceItem]


class LawChatResponse(BaseModel):
    ok: bool
    result: LawChatResult

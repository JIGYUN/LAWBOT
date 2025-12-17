# app/api/routes_law_chat.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.law_rag import search_law_with_answer, stream_law_answer
from app.rag.law_rag_graph import answer_law_question_graph, stream_law_answer_graph

router = APIRouter(prefix="/api/law", tags=["law"])


class LawChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class LawSource(BaseModel):
    id: str
    score: float
    dataset: str | None = None
    category: str | None = None
    question: str | None = None
    answerSnippet: str | None = None
    docNo: int | None = None


class LawChatResult(BaseModel):
    answer: str
    sources: List[LawSource]


class LawChatResponse(BaseModel):
    ok: bool
    result: LawChatResult


@router.post("/chat", response_model=LawChatResponse)
def chat_law(req: LawChatRequest) -> LawChatResponse:
    answer, points = search_law_with_answer(query=req.question, top_k=req.top_k)
    sources = [
        LawSource(
            id=p.get("id", ""),
            score=float(p.get("score", 0.0)),
            dataset=p.get("dataset"),
            category=p.get("category"),
            question=p.get("question"),
            answerSnippet=p.get("answerSnippet"),
            docNo=p.get("docNo"),
        )
        for p in points
    ]
    return LawChatResponse(ok=True, result=LawChatResult(answer=answer, sources=sources))


@router.post("/chat_stream")
def chat_law_stream(req: LawChatRequest) -> StreamingResponse:
    out = stream_law_answer(query=req.question, top_k=req.top_k)

    # (gen, points) 고정이지만, 혹시라도 변경되면 안전하게 흡수
    if isinstance(out, tuple):
        gen = out[0]
    else:
        gen = out

    return StreamingResponse(gen, media_type="text/plain; charset=utf-8")


@router.post("/chat_graph", response_model=LawChatResponse)
def chat_law_graph(req: LawChatRequest) -> LawChatResponse:
    answer, points = answer_law_question_graph(req.question, top_k=req.top_k)
    sources = [
        LawSource(
            id=p.get("id", ""),
            score=float(p.get("score", 0.0)),
            dataset=p.get("dataset"),
            category=p.get("category"),
            question=p.get("question"),
            answerSnippet=p.get("answerSnippet"),
            docNo=p.get("docNo"),
        )
        for p in points
    ]
    return LawChatResponse(ok=True, result=LawChatResult(answer=answer, sources=sources))


@router.post("/chat_graph_stream")
def chat_law_graph_stream(req: LawChatRequest) -> StreamingResponse:
    out = stream_law_answer_graph(req.question, top_k=req.top_k)

    # ✅ 여기서 “too many values” 절대 안 나게 처리
    # 기대: (gen, points)
    if not isinstance(out, tuple) or len(out) == 0:
        raise RuntimeError("stream_law_answer_graph() 반환값이 비정상입니다.")

    gen = out[0]
    # points는 out[1]에 있을 가능성이 높지만, stream은 우선 gen만 필요
    return StreamingResponse(gen, media_type="text/plain; charset=utf-8")

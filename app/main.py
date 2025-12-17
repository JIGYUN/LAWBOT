# filepath: app/main.py
from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_law_chat import router as law_chat_router
from app.ingestion.embeddings import embed_texts

app = FastAPI(title="LAWBOT API")

# React 등 다른 도메인에서 호출할 수 있도록 CORS 허용(필요 없으면 제거해도 됨)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 포트폴리오용이라 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_warmup() -> None:
    """
    sentence-transformers(KURE) 첫 encode 워밍업.
    --reload 재시작 시 첫 요청이 느려지는 현상 완화.
    """
    t0 = time.perf_counter()
    try:
        _ = embed_texts(["warmup"])
        ms = int((time.perf_counter() - t0) * 1000)
        print(f"[WARMUP] embeddings warmup ok ({ms} ms)")
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        print(f"[WARMUP] embeddings warmup failed ({ms} ms): {e}")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "status": "healthy"}


# 🔹 여기서 우리가 만든 법률 챗봇 라우터를 연결
app.include_router(law_chat_router)

# app/rag/law_rag_chain.py
from __future__ import annotations

from typing import Any, Dict, List

import httpx
from langchain_core.runnables import Runnable, RunnableLambda

from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_COLLECTION_OAI,
    QDRANT_TIMEOUT_SEC,
    EMBED_PROVIDER,
)
from app.ingestion.embeddings import embed_texts


def _active_collection_name() -> str:
    if (EMBED_PROVIDER or "").strip().lower() == "kure":
        return QDRANT_COLLECTION_NAME
    return QDRANT_COLLECTION_OAI or QDRANT_COLLECTION_NAME


def _embed_question(question: str) -> List[float]:
    vecs = embed_texts([question])
    if not vecs:
        return []
    return list(vecs[0])


def _http_search_points_by_vector(
    query_vec: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL 이 설정되어 있지 않습니다.")

    collection = _active_collection_name().strip()
    if not collection:
        raise RuntimeError("QDRANT_COLLECTION_NAME/QDRANT_COLLECTION_OAI 가 비었습니다.")

    base_url = QDRANT_URL.rstrip("/")
    url = f"{base_url}/collections/{collection}/points/search"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    body: Dict[str, Any] = {
        "vector": query_vec,
        "limit": int(top_k),
        "with_payload": True,
        "with_vector": False,
    }

    with httpx.Client(timeout=QDRANT_TIMEOUT_SEC) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("result", []) if isinstance(data, dict) else []
    points: List[Dict[str, Any]] = []
    for item in results:
        points.append(
            {
                "id": item.get("id", ""),
                "score": float(item.get("score") or 0.0),
                "payload": item.get("payload") or {},
                "collection": collection,
            }
        )
    return points


def build_retrieval_chain(top_k: int = 5) -> Runnable:
    def embed_step(question: str) -> List[float]:
        return _embed_question(question)

    def search_step(query_vec: List[float]) -> Dict[str, Any]:
        points = _http_search_points_by_vector(query_vec, top_k=top_k)
        return {"points": points}

    chain: Runnable = RunnableLambda(embed_step) | RunnableLambda(search_step)
    return chain

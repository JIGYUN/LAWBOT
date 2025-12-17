# app/ingestion/embeddings.py
from __future__ import annotations

import os
import time
from typing import List, Optional

from openai import OpenAI


def _embed_provider() -> str:
    # openai | kure
    return os.getenv("EMBED_PROVIDER", "openai").strip().lower()


# =========================
# OpenAI Embeddings
# =========================
def _get_oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY_LAWBOT") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY_LAWBOT 또는 OPENAI_API_KEY 환경 변수를 설정하세요.")
    return OpenAI(api_key=api_key)


def _oai_embed_model() -> str:
    return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()


def embed_texts_openai(
    texts: List[str],
    request_batch_size: int = 64,
    retry: int = 3,
) -> List[List[float]]:
    if not texts:
        return []

    client = _get_oai_client()
    model = _oai_embed_model()

    cleaned: List[str] = []
    for t in texts:
        s = (t or "").replace("\n", " ").strip()
        if s == "":
            s = " "
        cleaned.append(s)

    out: List[List[float]] = []

    for start in range(0, len(cleaned), request_batch_size):
        chunk = cleaned[start:start + request_batch_size]

        last_err: Exception | None = None
        for attempt in range(1, retry + 1):
            try:
                resp = client.embeddings.create(
                    model=model,
                    input=chunk,
                    encoding_format="float",
                )
                out.extend([d.embedding for d in resp.data])
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < retry:
                    time.sleep(0.8 * attempt)

        if last_err is not None:
            raise RuntimeError(f"OpenAI embeddings 실패: {last_err}") from last_err

    return out


# =========================
# KURE (lazy load)
# =========================
_KURE_MODEL: Optional[object] = None


def _get_kure_model():
    global _KURE_MODEL
    if _KURE_MODEL is not None:
        return _KURE_MODEL

    from sentence_transformers import SentenceTransformer  # lazy import

    model_name = os.getenv("KURE_MODEL_NAME", "nlpai-lab/KURE-v1")
    _KURE_MODEL = SentenceTransformer(model_name)
    return _KURE_MODEL


def embed_texts_kure(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    model = _get_kure_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


# =========================
# Unified
# =========================
def embed_texts(texts: List[str]) -> List[List[float]]:
    provider = _embed_provider()
    if provider == "kure":
        return embed_texts_kure(texts)
    return embed_texts_openai(texts)

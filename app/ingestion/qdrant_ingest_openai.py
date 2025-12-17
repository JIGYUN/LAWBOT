# app/ingestion/qdrant_ingest_openai.py
from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.ingestion.data_loader import LawRow


def _load_dotenv_once() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
            print(f"[ENV] loaded: {env_path}")
        else:
            print("[ENV] .env not found (skip)")
    except Exception as e:
        print(f"[ENV] dotenv load skipped: {e}")


_load_dotenv_once()


def _must_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"{name} 환경 변수를 설정하세요. (.env 로드 여부 확인)")
    return v


def get_qdrant_client() -> QdrantClient:
    qdrant_url = _must_env("QDRANT_URL")
    qdrant_api_key = _must_env("QDRANT_API_KEY")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60.0,
    )


def _get_oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY_LAWBOT") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY_LAWBOT 또는 OPENAI_API_KEY 환경 변수를 설정하세요.")
    return OpenAI(api_key=api_key)


def _oai_embed_model() -> str:
    return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def _oai_embed_dim() -> int:
    v = os.getenv("OPENAI_EMBED_DIM", "1536")
    try:
        return int(v)
    except ValueError as e:
        raise RuntimeError(f"OPENAI_EMBED_DIM 정수여야 합니다. value={v}") from e


def _qdrant_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION_OAI", "KLAC_BASIC_1_OAI_1536")


def _make_point_id(r: LawRow) -> str:
    """
    Qdrant point ID는 uint64 또는 UUID만 허용.
    → sha256(32bytes)에서 앞 16bytes를 써서 '결정적 UUID' 생성.
    """
    raw = f"{r.source_dataset}|{r.law_category}|{r.question}|{r.answer}"
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    # version=4로 UUID 비트만 정상화(결과는 raw 기반으로 항상 동일)
    return str(uuid.UUID(bytes=digest[:16], version=4))


def ensure_collection_exists() -> None:
    client = get_qdrant_client()
    collection = _qdrant_collection_name()
    dim = _oai_embed_dim()

    if client.collection_exists(collection):
        info = client.get_collection(collection)
        print(f"[QDRANT][OAI] 기존 컬렉션 존재: {collection} (points={info.points_count})")
        return

    print(f"[QDRANT][OAI] 컬렉션 없음, 새로 생성: {collection} (dim={dim})")
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print("[QDRANT][OAI] 컬렉션 생성 완료")


def ensure_payload_indexes() -> None:
    client = get_qdrant_client()
    collection = _qdrant_collection_name()
    fields = ["source_dataset", "law_category"]

    for f in fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=f,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            print(f"[QDRANT][OAI] payload index 생성: {f}")
        except Exception as e:
            print(f"[QDRANT][OAI] payload index skip(이미 존재 가능): {f} / {e}")


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
                    sleep_s = 0.8 * attempt
                    print(f"[OAI][EMBED] retry {attempt}/{retry} after {sleep_s:.1f}s: {e}")
                    time.sleep(sleep_s)

        if last_err is not None:
            raise RuntimeError(f"OpenAI embeddings 실패: {last_err}") from last_err

    return out


def upsert_law_rows(rows: List[LawRow], batch_size: int = 50) -> None:
    print(f"[QDRANT][OAI] upsert_law_rows 호출, rows={len(rows)}")

    if not rows:
        print("[QDRANT][OAI] upsert할 데이터가 없습니다.")
        return

    client = get_qdrant_client()
    ensure_collection_exists()
    ensure_payload_indexes()

    collection = _qdrant_collection_name()
    total = len(rows)

    print(f"[QDRANT][OAI] 총 {total}건 upsert 시작 (batch_size={batch_size})")
    print(f"[QDRANT][OAI] embed_model={_oai_embed_model()} embed_dim={_oai_embed_dim()} collection={collection}")

    for start in range(0, total, batch_size):
        chunk = rows[start:start + batch_size]

        texts = [f"{r.question}\n\n{r.answer}" for r in chunk]
        print(f"[QDRANT][OAI] 임베딩 계산 중... rows {start}~{start + len(chunk) - 1}")
        vectors = embed_texts_openai(texts)

        points: List[qmodels.PointStruct] = [
            qmodels.PointStruct(
                id=_make_point_id(r),
                vector=vec,
                payload={
                    "source_dataset": r.source_dataset,
                    "law_category": r.law_category,
                    "question": r.question,
                    "answer": r.answer,
                    "embed_model": _oai_embed_model(),
                },
            )
            for r, vec in zip(chunk, vectors)
        ]

        client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )

        end = start + len(chunk) - 1
        print(f"[QDRANT][OAI] upsert 완료: rows {start}~{end} ({len(chunk)}건)")

    print("[QDRANT][OAI] 전체 upsert 완료")

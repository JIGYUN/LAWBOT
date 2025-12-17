# app/ingestion/qdrant_ingest.py
from __future__ import annotations

from typing import List
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBED_DIM,
)
from app.ingestion.embeddings import embed_texts
from app.ingestion.data_loader import LawRow


def get_qdrant_client() -> QdrantClient:
    """
    Qdrant 클라이언트 생성.
    - 환경변수에서 URL / API KEY 를 읽는다.
    - timeout 을 넉넉하게 60초로 잡는다.
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY 환경 변수를 설정하세요.")

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0,  # write 타임아웃 대비
    )


def ensure_collection_exists() -> None:
    """
    law_qa_v1 컬렉션이 없으면 생성하고,
    있으면 현재 포인트 개수를 출력만 한다.
    """
    client = get_qdrant_client()

    if client.collection_exists(QDRANT_COLLECTION_NAME):
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        print(
            f"[QDRANT] 기존 컬렉션 존재: {QDRANT_COLLECTION_NAME} "
            f"(points={info.points_count})"
        )
        return

    print(
        f"[QDRANT] 컬렉션 없음, 새로 생성: "
        f"{QDRANT_COLLECTION_NAME} (dim={EMBED_DIM})"
    )
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print("[QDRANT] 컬렉션 생성 완료")


def upsert_law_rows(rows: List[LawRow], batch_size: int = 50) -> None:
    """
    LawRow 리스트를 임베딩 → Qdrant 컬렉션에 upsert.

    - 한 번에 너무 많은 포인트를 보내면 Qdrant Cloud에서 write timeout 날 수 있으므로
      batch_size 기본값을 50으로 작게 잡는다.
    - wait=True 로 실제 쓰기가 끝날 때까지 기다린다.
    """
    # ✅ 여기서 바로 rows 길이 찍어서, 함수가 호출됐는지부터 확인
    print(f"[QDRANT] upsert_law_rows 호출, rows={len(rows)}")

    if not rows:
        print("[QDRANT] upsert할 데이터가 없습니다.")
        return

    client = get_qdrant_client()
    ensure_collection_exists()

    total = len(rows)
    print(f"[QDRANT] 총 {total}건 upsert 시작 (batch_size={batch_size})")

    for start in range(0, total, batch_size):
        chunk = rows[start:start + batch_size]

        # 1) 텍스트 결합 후 임베딩
        texts = [f"{r.question}\n\n{r.answer}" for r in chunk]
        print(f"[QDRANT] 임베딩 계산 중... rows {start}~{start + len(chunk) - 1}")
        vectors = embed_texts(texts)

        # 2) Qdrant PointStruct 생성
        points: List[qmodels.PointStruct] = [
            qmodels.PointStruct(
                id=str(uuid4()),
                vector=vec,
                payload={
                    "source_dataset": r.source_dataset,
                    "law_category": r.law_category,
                    "question": r.question,
                    "answer": r.answer,
                },
            )
            for r, vec in zip(chunk, vectors)
        ]

        # 3) 업서트 (wait=True 로 실제 기록 완료까지 대기)
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
            wait=True,
        )

        end = start + len(chunk) - 1
        print(f"[QDRANT] upsert 완료: rows {start}~{end} ({len(chunk)}건)")

    print("[QDRANT] 전체 upsert 완료")

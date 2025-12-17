# app/rag/law_rag.py
from __future__ import annotations

from typing import Any, Dict, Generator, List, Tuple
import json
import time

import requests
from openai import OpenAI

from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,      # 레거시(KURE)
    QDRANT_COLLECTION_OAI,       # 신규(OpenAI)
    QDRANT_TIMEOUT_SEC,
    EMBED_PROVIDER,
    OPENAI_API_KEY_LAWBOT,
    OPENAI_LAWBOT_MODEL,
    OPENAI_BASE_URL,
    OPENAI_TIMEOUT_SEC,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
)
from app.ingestion.embeddings import embed_texts


# -----------------------------
# HTTP session (keep-alive)
# -----------------------------
_QDRANT_SESSION = requests.Session()


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _active_collection_name() -> str:
    """
    임베딩 제공자(EMBED_PROVIDER)에 따라 조회 컬렉션을 자동 선택.
    - openai -> QDRANT_COLLECTION_OAI
    - kure   -> QDRANT_COLLECTION_NAME
    """
    if (EMBED_PROVIDER or "").strip().lower() == "kure":
        return QDRANT_COLLECTION_NAME
    return QDRANT_COLLECTION_OAI or QDRANT_COLLECTION_NAME


def _qdrant_headers() -> Dict[str, str]:
    h: Dict[str, str] = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h


def qdrant_search_laws(
    query_vec: List[float],
    top_k: int = 5,
    collection_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    qdrant-client 버전 차이로 메서드가 깨지는 문제를 피하려고
    ✅ REST API(points/search)로 고정.
    """
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL 이 비었습니다. .env 확인")

    col = (collection_name or _active_collection_name()).strip()
    if not col:
        raise RuntimeError("QDRANT_COLLECTION_NAME/QDRANT_COLLECTION_OAI 가 비었습니다. .env 확인")

    url = f"{QDRANT_URL.rstrip('/')}/collections/{col}/points/search"
    payload = {
        "vector": query_vec,
        "limit": int(top_k),
        "with_payload": True,
        "with_vector": False,
    }

    resp = _QDRANT_SESSION.post(
        url,
        headers=_qdrant_headers(),
        data=json.dumps(payload),
        timeout=QDRANT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("result", [])
    out: List[Dict[str, Any]] = []

    for r in results:
        payload_obj = r.get("payload", {}) or {}

        dataset = (
            payload_obj.get("dataset_name")
            or payload_obj.get("dataset")
            or payload_obj.get("source_dataset")
        )
        category = payload_obj.get("law_category") or payload_obj.get("category")
        question = payload_obj.get("question")
        answer = payload_obj.get("answer")

        out.append(
            {
                "id": str(r.get("id", "")),
                "score": float(r.get("score", 0.0)),
                "dataset": dataset,
                "category": category,
                "question": question,
                "answer": answer,
                "answerSnippet": None,
                "docNo": None,
                "collection": col,
            }
        )

    return out


def build_context_from_points(points: List[Dict[str, Any]], max_chars_per_doc: int = 900) -> str:
    lines: List[str] = []
    for i, p in enumerate(points, start=1):
        q = (p.get("question") or "").strip()
        a = (p.get("answer") or "").strip()

        if max_chars_per_doc > 0 and len(a) > max_chars_per_doc:
            a = a[:max_chars_per_doc] + "…"

        lines.append(f"[문서 {i}]")
        if q:
            lines.append(f"Q: {q}")
        if a:
            lines.append(f"A: {a}")
        lines.append("")

    return "\n".join(lines).strip()


def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY_LAWBOT:
        raise RuntimeError("OPENAI_API_KEY_LAWBOT 이 비었습니다. .env 확인")

    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY_LAWBOT, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY_LAWBOT)


def call_openai_chat(system_prompt: str, user_prompt: str) -> str:
    client = _openai_client()
    resp = client.chat.completions.create(
        model=OPENAI_LAWBOT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(OPENAI_TEMPERATURE),
        max_tokens=int(OPENAI_MAX_TOKENS),
        timeout=float(OPENAI_TIMEOUT_SEC),
    )
    return (resp.choices[0].message.content or "").strip()


def call_openai_chat_stream(system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
    client = _openai_client()
    t0 = _now_ms()
    first = True

    stream = client.chat.completions.create(
        model=OPENAI_LAWBOT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(OPENAI_TEMPERATURE),
        max_tokens=int(OPENAI_MAX_TOKENS),
        timeout=float(OPENAI_TIMEOUT_SEC),
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if not delta:
            continue

        if first:
            first = False
            print(f"[OPENAI STREAM] first_token_ms={_now_ms() - t0}")

        yield delta


def _build_prompts(question: str, context: str) -> Tuple[str, str]:
    system_prompt = (
        "당신은 한국어 법률 상담 보조 AI입니다.\n"
        "반드시 제공된 [문서 n] 근거를 우선하여 답하고, 근거가 부족하면 '추가상담 필요'를 명시하세요.\n"
        "출력 형식:\n"
        "1) 요약\n"
        "2) 단계\n"
        "3) 주의사항(리스크)\n"
        "4) 추가상담\n"
    )

    user_prompt = (
        f"질문: {question}\n\n"
        f"아래는 참고 문서(근거)입니다:\n{context}\n\n"
        "요구 형식대로 답해주세요. 각 문장은 가능한 한 [문서 n] 근거를 붙이세요."
    )
    return system_prompt, user_prompt


def search_law_with_answer(query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    rid = 1
    t_all = _now_ms()
    active_col = _active_collection_name()

    # 1) embed
    t0 = _now_ms()
    query_vec = embed_texts([query])[0]
    embed_ms = _now_ms() - t0

    # 2) qdrant
    t1 = _now_ms()
    points = qdrant_search_laws(query_vec, top_k=int(top_k), collection_name=active_col)
    qdrant_ms = _now_ms() - t1

    # 3) prompt
    context = build_context_from_points(points)
    system_prompt, user_prompt = _build_prompts(query, context)

    prompt_chars = len(system_prompt) + len(user_prompt)
    context_chars = len(context)

    # 4) openai chat
    t2 = _now_ms()
    answer = call_openai_chat(system_prompt, user_prompt)
    openai_ms = _now_ms() - t2

    total_ms = _now_ms() - t_all

    print("\n========== [LAW RAG TIMING] ==========")
    print(f"- rid          : {rid}")
    print(f"- provider     : {EMBED_PROVIDER}")
    print(f"- collection   : {active_col}")
    print(f"- top_k        : {top_k}")
    print(f"- emb_dim      : {len(query_vec)}")
    print(f"- points_count : {len(points)}")
    print(f"- context_chars: {context_chars}")
    print(f"- prompt_chars : {prompt_chars}")
    print(f"- embed_ms     : {embed_ms}")
    print(f"- qdrant_ms    : {qdrant_ms}")
    print(f"- openai_ms    : {openai_ms}")
    print(f"- total_ms     : {total_ms}")
    print("=====================================\n")

    return answer, points


def stream_law_answer(query: str, top_k: int = 5) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
    rid = 1
    t_all = _now_ms()
    active_col = _active_collection_name()

    # 1) embed
    t0 = _now_ms()
    query_vec = embed_texts([query])[0]
    embed_ms = _now_ms() - t0

    # 2) qdrant
    t1 = _now_ms()
    points = qdrant_search_laws(query_vec, top_k=int(top_k), collection_name=active_col)
    qdrant_ms = _now_ms() - t1

    # 3) prompt (prep)
    context = build_context_from_points(points)
    system_prompt, user_prompt = _build_prompts(query, context)

    prompt_chars = len(system_prompt) + len(user_prompt)
    context_chars = len(context)

    prep_total_ms = _now_ms() - t_all

    print("\n========== [LAW RAG STREAM PREP] ==========")
    print(f"- rid          : {rid}")
    print(f"- provider     : {EMBED_PROVIDER}")
    print(f"- collection   : {active_col}")
    print(f"- top_k        : {top_k}")
    print(f"- emb_dim      : {len(query_vec)}")
    print(f"- points_count : {len(points)}")
    print(f"- context_chars: {context_chars}")
    print(f"- prompt_chars : {prompt_chars}")
    print(f"- embed_ms     : {embed_ms}")
    print(f"- qdrant_ms    : {qdrant_ms}")
    print(f"- prep_total_ms: {prep_total_ms}")
    print("==========================================\n")

    gen = call_openai_chat_stream(system_prompt, user_prompt)
    return gen, points

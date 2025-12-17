# app/rag/law_rag_graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict

import time

from langgraph.graph import StateGraph, END

from app.ingestion.embeddings import embed_texts
from app.rag.law_rag import (
    build_context_from_points,
    call_openai_chat,
    call_openai_chat_stream,
    qdrant_search_laws,
)


class GraphState(TypedDict, total=False):
    question: str
    top_k: int

    # timing
    embed_ms: int
    qdrant_ms: int

    # retrieval
    query_vec: List[float]
    points: List[Dict[str, Any]]
    context: str

    # prompt
    system_prompt: str
    user_prompt: str

    # output
    answer: str


def _node_embed(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    q = state["question"]
    vec = embed_texts([q])[0]
    dt = int((time.perf_counter() - t0) * 1000)
    state["query_vec"] = vec
    state["embed_ms"] = dt
    return state


def _node_retrieve(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    vec = state["query_vec"]
    top_k = int(state.get("top_k", 5))
    pts = qdrant_search_laws(vec, top_k=top_k)
    dt = int((time.perf_counter() - t0) * 1000)
    state["points"] = pts
    state["qdrant_ms"] = dt
    state["context"] = build_context_from_points(pts)
    return state


def _node_prompt(state: GraphState) -> GraphState:
    q = state["question"]
    context = state.get("context", "")

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
        f"질문: {q}\n\n"
        f"아래는 참고 문서(근거)입니다:\n{context}\n\n"
        "요구 형식대로 답해주세요. 각 문장은 가능한 한 [문서 n] 근거를 붙이세요."
    )

    state["system_prompt"] = system_prompt
    state["user_prompt"] = user_prompt
    return state


def _node_generate(state: GraphState) -> GraphState:
    system_prompt = state["system_prompt"]
    user_prompt = state["user_prompt"]
    answer = call_openai_chat(system_prompt, user_prompt)
    state["answer"] = answer
    return state


def _build_graph_full() -> Any:
    g = StateGraph(GraphState)
    g.add_node("embed", _node_embed)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("prompt", _node_prompt)
    g.add_node("generate", _node_generate)

    g.set_entry_point("embed")
    g.add_edge("embed", "retrieve")
    g.add_edge("retrieve", "prompt")
    g.add_edge("prompt", "generate")
    g.add_edge("generate", END)
    return g.compile()


def _build_graph_prep() -> Any:
    # stream용: generate 없이 prompt까지만
    g = StateGraph(GraphState)
    g.add_node("embed", _node_embed)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("prompt", _node_prompt)

    g.set_entry_point("embed")
    g.add_edge("embed", "retrieve")
    g.add_edge("retrieve", "prompt")
    g.add_edge("prompt", END)
    return g.compile()


LAW_GRAPH_FULL = _build_graph_full()
LAW_GRAPH_PREP = _build_graph_prep()


def answer_law_question_graph(question: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    rid = 1
    t_all = time.perf_counter()

    state: GraphState = {"question": question, "top_k": int(top_k)}
    t0 = time.perf_counter()
    final: GraphState = LAW_GRAPH_FULL.invoke(state)
    total_ms = int((time.perf_counter() - t_all) * 1000)

    points = final.get("points", [])
    context = final.get("context", "")
    system_prompt = final.get("system_prompt", "")
    user_prompt = final.get("user_prompt", "")
    answer = final.get("answer", "")

    prompt_chars = len(system_prompt) + len(user_prompt)
    context_chars = len(context)

    embed_ms = int(final.get("embed_ms", 0))
    qdrant_ms = int(final.get("qdrant_ms", 0))

    # openai_ms는 graph 내부에서 측정이 어려워서 "전체 - (embed+qdrant+prompt)"로 추정하지 않고,
    # 여기선 로깅을 total로만 깔끔히 남긴다. (원하면 node_generate 안에서 측정 가능)
    print("\n========== [LAW GRAPH TIMING] ==========")
    print(f"- rid          : {rid}")
    print(f"- top_k        : {top_k}")
    print(f"- emb_dim      : {len(final.get('query_vec', []))}")
    print(f"- points_count : {len(points)}")
    print(f"- context_chars: {context_chars}")
    print(f"- prompt_chars : {prompt_chars}")
    print(f"- embed_ms     : {embed_ms}")
    print(f"- qdrant_ms    : {qdrant_ms}")
    print(f"- total_ms     : {total_ms}")
    print("=======================================\n")

    return answer, points


def stream_law_answer_graph(question: str, top_k: int = 5) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    ✅ 반드시 (gen, points) 2개만 리턴한다.
    라우터에서 gen, points = ... 로 안정적으로 받을 수 있게 고정.
    """
    rid = 1
    t_all = time.perf_counter()

    state: GraphState = {"question": question, "top_k": int(top_k)}
    final: GraphState = LAW_GRAPH_PREP.invoke(state)

    points = final.get("points", [])
    context = final.get("context", "")
    system_prompt = final.get("system_prompt", "")
    user_prompt = final.get("user_prompt", "")

    embed_ms = int(final.get("embed_ms", 0))
    qdrant_ms = int(final.get("qdrant_ms", 0))
    context_chars = len(context)
    prompt_chars = len(system_prompt) + len(user_prompt)

    prep_total_ms = int((time.perf_counter() - t_all) * 1000)

    print("\n========== [LAW GRAPH STREAM PREP] ==========")
    print(f"- rid          : {rid}")
    print(f"- top_k        : {top_k}")
    print(f"- emb_dim      : {len(final.get('query_vec', []))}")
    print(f"- points_count : {len(points)}")
    print(f"- context_chars: {context_chars}")
    print(f"- prompt_chars : {prompt_chars}")
    print(f"- embed_ms     : {embed_ms}")
    print(f"- qdrant_ms    : {qdrant_ms}")
    print(f"- prep_total_ms: {prep_total_ms}")
    print("=============================================\n")

    gen = call_openai_chat_stream(system_prompt, user_prompt)
    return gen, points

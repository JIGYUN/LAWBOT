from __future__ import annotations

import os
import statistics
import time
from dotenv import load_dotenv

load_dotenv()

from app.ingestion.embeddings import embed_texts  # noqa: E402


SAMPLES = [
    "집주인이 전세보증금을 안 돌려줘요. 어떻게 해야 하나요?",
    "임차권등기명령 신청 절차가 궁금합니다.",
    "내용증명은 꼭 보내야 하나요?",
    "보증금 반환 소송을 하면 기간이 얼마나 걸리나요?",
    "전세금반환보증 가입 조건이 어떻게 되나요?",
]


def now_ms() -> int:
    return int(time.perf_counter() * 1000)


def main() -> None:
    n = int(os.getenv("BENCH_N", "20"))
    provider = os.getenv("EMBED_PROVIDER", "openai")
    print(f"[BENCH] EMBED_PROVIDER={provider}  N={n}")

    # warmup
    embed_texts([SAMPLES[0]])

    times: list[int] = []
    for i in range(n):
        t0 = now_ms()
        _ = embed_texts([SAMPLES[i % len(SAMPLES)]])[0]
        dt = now_ms() - t0
        times.append(dt)

    print(f"[BENCH] ms: min={min(times)}  p50={int(statistics.median(times))}  avg={int(statistics.mean(times))}  max={max(times)}")


if __name__ == "__main__":
    main()

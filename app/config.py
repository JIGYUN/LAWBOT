# app/config.py
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# =============================
# Qdrant
# =============================
QDRANT_URL: str = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "").strip()

# 기존(KURE) 컬렉션 (레거시)
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "law_qa_v1").strip()

# 신규(OpenAI) 컬렉션
QDRANT_COLLECTION_OAI: str = os.getenv("QDRANT_COLLECTION_OAI", "KLAC_BASIC_1_OAI_1536").strip()

QDRANT_TIMEOUT_SEC: float = float(os.getenv("QDRANT_TIMEOUT_SEC", "30"))

# =============================
# Embeddings Provider
# =============================
# openai | kure
EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "openai").strip().lower()

# KURE-v1 임베딩 차원(레거시)
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1024"))

# OpenAI 임베딩 모델/차원
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
OPENAI_EMBED_DIM: int = int(os.getenv("OPENAI_EMBED_DIM", "1536"))

# =============================
# OpenAI (LAWBOT - Chat)
# =============================
OPENAI_API_KEY_LAWBOT: str = os.getenv("OPENAI_API_KEY_LAWBOT", "").strip()
OPENAI_LAWBOT_MODEL: str = os.getenv("OPENAI_LAWBOT_MODEL", "gpt-4.1-mini").strip()

# 선택: 프록시/게이트웨이 쓰는 경우에만 설정
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "").strip()

# 타임아웃/생성 옵션
OPENAI_TIMEOUT_SEC: float = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "700"))

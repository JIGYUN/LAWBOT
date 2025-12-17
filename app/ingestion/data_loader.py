# app/ingestion/data_loader.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

# 루트(LAWBOT) 밑의 data 폴더를 기준으로 사용
DATA_DIR = Path("data")


@dataclass
class LawRow:
    source_dataset: str
    law_category: str
    question: str
    answer: str


def _read_csv(path: Path) -> Iterable[dict]:
    """
    공통 CSV 로더.
    data.go.kr 파일이 보통 EUC-KR(=cp949)이어서 그 인코딩으로 읽는다.
    """
    with path.open("r", encoding="cp949", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_basic_1() -> List[LawRow]:
    """
    대한법률구조공단_법률상담 기본질문답변_1차 하나만 로딩.

    ⚠ CSV 컬럼명은 실제 파일을 열어서 확인해야 한다.
       일반적으로는 다음과 같이 되어 있음:
         - 법률분류
         - 기본질문
         - 기본답변
    """
    # 파일 이름을 실제 data 폴더의 이름과 맞춰줄 것.
    # 예) data/KLAC_BASIC_1.csv
    path = DATA_DIR / "KLAC_BASIC_1.csv"

    rows: List[LawRow] = []

    for raw in _read_csv(path):
        law_category = (raw.get("법률분류") or "").strip()
        question = (raw.get("기본질문") or "").strip()
        answer = (raw.get("기본답변") or "").strip()

        if not question or not answer:
            continue

        rows.append(
            LawRow(
                source_dataset="KLAC_BASIC_1",
                law_category=law_category,
                question=question,
                answer=answer,
            )
        )

    return rows


def load_all_law_rows() -> List[LawRow]:
    """
    현재는 BASIC_1 하나만 사용.
    나중에 BASIC_2 / CASE_1 등을 추가하고 싶으면 여기서 extend 하면 됨.
    """
    return load_basic_1()

# app/ingestion/ingest_oai_cli.py
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from app.ingestion.data_loader import LawRow
from app.ingestion.qdrant_ingest_openai import upsert_law_rows


def _pick(d: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            s = str(v).strip()
            if s != "":
                return s
    return None


def _default_source_dataset(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    return name or "dataset"


def read_law_rows_from_csv(csv_path: str, encoding: str) -> List[LawRow]:
    rows: List[LawRow] = []
    source_default = os.getenv("SOURCE_DATASET", _default_source_dataset(csv_path))

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            # 다양한 컬럼명 대응(너희 CSV/로더가 바뀌어도 최대한 살아남게)
            source_dataset = _pick(r, ["source_dataset", "sourceDataset", "dataset_name", "dataset", "datasetName"]) or source_default
            law_category = _pick(r, ["law_category", "lawCategory", "category", "법률분류"]) or "UNKNOWN"
            question = _pick(r, ["question", "q", "기본질문"]) or ""
            answer = _pick(r, ["answer", "a", "기본답변"]) or ""

            if question.strip() == "" or answer.strip() == "":
                # 빈 줄/깨진 데이터는 스킵
                continue

            rows.append(
                LawRow(
                    source_dataset=source_dataset,
                    law_category=law_category,
                    question=question,
                    answer=answer,
                )
            )

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV 파일 경로")
    p.add_argument("--encoding", default="cp949", help="CSV 인코딩 (기본 cp949)")
    p.add_argument("--batch-size", type=int, default=50, help="Qdrant upsert batch size (기본 50)")
    args = p.parse_args()

    csv_path = args.csv
    rows = read_law_rows_from_csv(csv_path, args.encoding)

    print(f"[INGEST][OAI] loaded rows={len(rows)} from {csv_path} (encoding={args.encoding})")

    # 미리 3개만 프린트(데이터 확인)
    for i, r in enumerate(rows[:3]):
        print(f"[INGEST][OAI] sample[{i}]={asdict(r)}")

    upsert_law_rows(rows, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

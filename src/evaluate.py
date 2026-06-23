"""Run final pipeline over evaluation questions and save summary/doc-level CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from pipeline import run_full_pipeline


DEFAULT_QUESTIONS = [
    "어린이 보호구역에서는 몇 km로 달려야 해?",
    "스쿨존에서 속도위반하면 어떻게 돼?",
    "주정차 위반하면 과태료가 얼마나 나와?",
    "음주운전하면 면허가 바로 취소돼?",
    "불법주차한 차는 신고할 수 있어?",
    "주차위반한 차는 바로 견인될 수 있어?",
    "운전하면서 휴대폰 보면 처벌받아?",
    "음주측정 거부하면 어떻게 돼?",
    "횡단보도에서 보행자는 얼마나 보호받아?",
    "무면허운전하면 처벌이 어떻게 돼?",
]


def evaluate(questions: Iterable[str] = DEFAULT_QUESTIONS, output_dir: str = "outputs") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    doc_rows = []

    for query in questions:
        result = run_full_pipeline(query=query, top_n=5)
        summary_rows.append({
            "query": query,
            "query_tokens": ", ".join(result["query_tokens"]),
            "answer": result["answer"],
            "num_final_docs": len(result["final_docs"]),
            "context_preview": result["context"][:1500],
        })
        for doc in result["final_docs"]:
            doc_rows.append({
                "query": query,
                "query_tokens": ", ".join(result["query_tokens"]),
                "final_rank": doc["rank"],
                "rerank_score": doc["rerank_score"],
                "rrf_score": doc["rrf_score"],
                "rrf_rank": doc["rrf_rank"],
                "dense_rank": doc["dense_rank"],
                "bm25_rank": doc["bm25_rank"],
                "metadata": doc["metadata"],
                "page_content": doc["page_content"],
            })

    pd.DataFrame(summary_rows).to_csv(output_path / "final_pipeline_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(doc_rows).to_csv(output_path / "final_pipeline_docs.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    evaluate()

"""End-to-end retrieval and generation pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from generator import call_llm_with_context
from hybrid_search import get_rrf_candidates
from reranker import rerank_documents


def serialize_docs(docs) -> List[Dict[str, Any]]:
    return [
        {"rank": idx, "metadata": doc.metadata, "page_content": doc.page_content}
        for idx, doc in enumerate(docs, start=1)
    ]


def run_retrieval_pipeline(query: str) -> Dict[str, Any]:
    retrieval = get_rrf_candidates(query)
    reranked_docs = rerank_documents(query=query, rrf_docs=retrieval["rrf_docs"])

    return {
        "query": query,
        "query_tokens": retrieval["query_tokens"],
        "dense": serialize_docs(retrieval["dense_results"]),
        "bm25": serialize_docs(retrieval["bm25_results"]),
        "rrf": [
            {
                "rank": item["rrf_rank"],
                "rrf_score": item["rrf_score"],
                "dense_rank": item["dense_rank"],
                "bm25_rank": item["bm25_rank"],
                "metadata": item["doc"].metadata,
                "page_content": item["doc"].page_content,
            }
            for item in retrieval["rrf_docs"]
        ],
        "reranked": [
            {
                "rank": idx,
                "rerank_score": item["rerank_score"],
                "rrf_score": item["rrf_score"],
                "rrf_rank": item["rrf_rank"],
                "dense_rank": item["dense_rank"],
                "bm25_rank": item["bm25_rank"],
                "metadata": item["doc"].metadata,
                "page_content": item["doc"].page_content,
            }
            for idx, item in enumerate(reranked_docs, start=1)
        ],
    }


def get_final_context(query: str, top_n: int = 5) -> Dict[str, Any]:
    result = run_retrieval_pipeline(query)
    final_docs = result["reranked"][:top_n]
    context = "\n\n".join(doc["page_content"] for doc in final_docs)
    return {
        "query": query,
        "query_tokens": result["query_tokens"],
        "final_docs": final_docs,
        "context": context,
    }


def run_full_pipeline(query: str, top_n: int = 5, temperature: float = 0.1, max_tokens: int = 160) -> Dict[str, Any]:
    retrieval = get_final_context(query, top_n=top_n)
    answer = call_llm_with_context(
        query=query,
        context=retrieval["context"],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {**retrieval, "answer": answer}


def ask(query: str) -> str:
    return run_full_pipeline(query=query)["answer"]


if __name__ == "__main__":
    question = input("질문을 입력하세요: ").strip()
    print(ask(question))

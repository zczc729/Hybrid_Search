"""Dense + BM25 + RRF hybrid search."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from bm25_retriever import build_bm25_retriever, custom_preprocess
from config import RRF_K, TOP_K_RRF
from dense_retriever import get_dense_retriever, load_all_docs_from_chroma


def _doc_key(doc: Document) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    return (
        doc.page_content.strip(),
        tuple(sorted((doc.metadata or {}).items())),
    )


def reciprocal_rank_fusion(
    dense_results: List[Document],
    bm25_results: List[Document],
    k: int = RRF_K,
    top_k: int = TOP_K_RRF,
) -> List[Dict[str, Any]]:
    scores = defaultdict(float)
    doc_store: Dict[Any, Document] = {}
    dense_rank_map: Dict[Any, int] = {}
    bm25_rank_map: Dict[Any, int] = {}

    for rank, doc in enumerate(dense_results, start=1):
        key = _doc_key(doc)
        scores[key] += 1 / (k + rank)
        doc_store[key] = doc
        dense_rank_map[key] = rank

    for rank, doc in enumerate(bm25_results, start=1):
        key = _doc_key(doc)
        scores[key] += 1 / (k + rank)
        doc_store[key] = doc
        bm25_rank_map[key] = rank

    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            "doc_key": key,
            "doc": doc_store[key],
            "rrf_score": score,
            "rrf_rank": idx,
            "dense_rank": dense_rank_map.get(key),
            "bm25_rank": bm25_rank_map.get(key),
        }
        for idx, (key, score) in enumerate(sorted_items, start=1)
    ]


def get_rrf_candidates(query: str) -> Dict[str, Any]:
    dense_retriever = get_dense_retriever()
    all_docs = load_all_docs_from_chroma()
    bm25_retriever = build_bm25_retriever(all_docs)

    dense_results = dense_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)
    rrf_docs = reciprocal_rank_fusion(dense_results, bm25_results)

    return {
        "query": query,
        "query_tokens": custom_preprocess(query),
        "dense_results": dense_results,
        "bm25_results": bm25_results,
        "rrf_docs": rrf_docs,
    }

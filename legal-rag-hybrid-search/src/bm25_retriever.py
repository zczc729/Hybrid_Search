"""BM25 retriever and Korean legal-domain query preprocessing."""

from __future__ import annotations

import re
from typing import List

from konlpy.tag import Okt
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import COMPOUND_TERMS, STOPWORDS, TOP_K_BM25

_okt = Okt()
_COMPOUND_TERMS_NORMALIZED = [term.replace(" ", "") for term in COMPOUND_TERMS]


def custom_preprocess(text: str) -> List[str]:
    """Tokenize Korean queries for BM25 while preserving legal-domain compound terms."""
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    pos_tags = _okt.pos(text, norm=True, stem=True)

    base_tokens: List[str] = []
    for word, pos in pos_tags:
        if pos in {"Noun", "Alpha", "Number"} and len(word) >= 2:
            base_tokens.append(word)

    compact_text = re.sub(r"\s+", "", text)
    compound_tokens = [term for term in _COMPOUND_TERMS_NORMALIZED if term in compact_text]

    cleaned: List[str] = []
    seen = set()
    for token in base_tokens + compound_tokens:
        token = token.strip()
        if not token or token in STOPWORDS or len(token) < 2:
            continue
        if token not in seen:
            seen.add(token)
            cleaned.append(token)
    return cleaned


def build_bm25_retriever(docs: List[Document], top_k: int = TOP_K_BM25) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs, preprocess_func=custom_preprocess)
    retriever.k = top_k
    return retriever

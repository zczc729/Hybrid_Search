"""Rerank RRF candidates with a cross-encoder reranker."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, RERANKER_MODEL, TOP_K_FINAL


def get_torch_device() -> torch.device:
    if DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.device = get_torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: List[str], batch_size: int = 8, max_length: int = 512) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start:start + batch_size]
            pairs = [[query, passage] for passage in batch]
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits.view(-1).float().cpu().tolist()
                scores.extend(logits)
        return scores


def rerank_documents(query: str, rrf_docs: List[Dict[str, Any]], top_k_final: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    reranker = Reranker()
    passages = [item["doc"].page_content for item in rrf_docs]
    scores = reranker.score(query, passages)

    results = []
    for item, score in zip(rrf_docs, scores):
        copied = dict(item)
        copied["rerank_score"] = score
        results.append(copied)

    return sorted(results, key=lambda item: item["rerank_score"], reverse=True)[:top_k_final]

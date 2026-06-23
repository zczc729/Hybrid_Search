"""Build Chroma vector DB from preprocessed legal documents."""

from __future__ import annotations

from typing import List

import chromadb
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DEVICE, EMBEDDING_MODEL, PERSIST_DIR, RAW_PDF_DIR
from preprocessing import load_and_split_pdfs


class SentenceTransformerEmbedding:
    """LangChain-compatible wrapper for SentenceTransformer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def build_vectorstore(pdf_dir: str = RAW_PDF_DIR, persist_dir: str = PERSIST_DIR) -> Chroma:
    docs = load_and_split_pdfs(pdf_dir)
    embedding_fn = SentenceTransformerEmbedding()
    client = chromadb.PersistentClient(path=persist_dir)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    vectorstore.add_documents(docs)
    return vectorstore


if __name__ == "__main__":
    vs = build_vectorstore()
    print("Vector DB build completed.")

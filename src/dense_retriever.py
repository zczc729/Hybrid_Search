"""Dense retriever loader for Chroma."""

from __future__ import annotations

from typing import List

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import COLLECTION_NAME, PERSIST_DIR, TOP_K_DENSE
from embedding import SentenceTransformerEmbedding


def load_vectorstore(persist_dir: str = PERSIST_DIR) -> Chroma:
    client = chromadb.PersistentClient(path=persist_dir)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbedding(),
    )


def get_dense_retriever(top_k: int = TOP_K_DENSE):
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def load_all_docs_from_chroma() -> List[Document]:
    vectorstore = load_vectorstore()
    raw = vectorstore.get(include=["documents", "metadatas"])
    docs: List[Document] = []
    for content, metadata in zip(raw.get("documents", []), raw.get("metadatas", [])):
        if content and str(content).strip():
            docs.append(Document(page_content=content, metadata=metadata or {}))
    return docs

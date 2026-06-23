"""PDF preprocessing utilities for Korean legal documents."""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from tqdm import tqdm

from config import RAW_PDF_DIR


def clean_legal_text(text: str) -> str:
    """Remove repetitive headers, footers, page numbers and noisy legal PDF artifacts."""
    text = re.sub(r"^\s*법제처.*국가법령정보센터\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\(.*?\)\s*\d{2,3}-\d{3,4}-\d{4}", "", text)
    text = re.sub(r"^\s*\[.*?\]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*제\s*\d+\s*(장|절|관)\b.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\n\s*){2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def infer_law_title(pdf_path: str | Path) -> str:
    """Infer a law title from a PDF filename."""
    stem = Path(pdf_path).stem
    stem = re.sub(r"\(.*?\)", "", stem)
    stem = re.sub(r"\d{8}$", "", stem).strip()
    return stem or Path(pdf_path).stem


def split_by_article(text: str, law_title: str, source: str) -> List[Document]:
    """Split legal text by article headings such as 제12조 or 제12조의2."""
    pattern = re.compile(r"(?=제\s*\d+\s*조(?:의\s*\d+)?\s*\()")
    parts = [p.strip() for p in pattern.split(text) if p.strip()]
    docs: List[Document] = []

    for part in parts:
        article_match = re.search(r"제\s*\d+\s*조(?:의\s*\d+)?", part)
        article = re.sub(r"\s+", "", article_match.group()) if article_match else ""
        content = f"{law_title} {part}".strip()
        if len(content) < 30:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "law_title": law_title,
                    "article": article,
                    "source": str(source),
                },
            )
        )
    return docs


def load_and_split_pdfs(pdf_dir: str | Path = RAW_PDF_DIR, max_files: Optional[int] = None) -> List[Document]:
    """Load PDFs from a directory and return article-level Documents."""
    pdf_paths = sorted(glob(str(Path(pdf_dir) / "*.pdf")))
    if max_files is not None:
        pdf_paths = pdf_paths[:max_files]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    all_docs: List[Document] = []
    for path in tqdm(pdf_paths, desc="Loading PDFs"):
        law_title = infer_law_title(path)
        pages = PyMuPDFLoader(path).load()
        raw_text = "\n".join(page.page_content for page in pages)
        cleaned = clean_legal_text(raw_text)
        all_docs.extend(split_by_article(cleaned, law_title=law_title, source=path))
    return all_docs


if __name__ == "__main__":
    docs = load_and_split_pdfs()
    print(f"Loaded {len(docs):,} article-level documents")
    if docs:
        print(docs[0].metadata)
        print(docs[0].page_content[:500])

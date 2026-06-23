"""Project configuration.

Environment variables can override the defaults below.
This file intentionally avoids hard-coded personal paths or private keys.
"""

import os

# Paths
RAW_PDF_DIR = os.getenv("RAW_PDF_DIR", "data/raw")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "laws_by_article")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Alibaba-NLP/gte-multilingual-reranker-base")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")

# Runtime
DEVICE = os.getenv("DEVICE", "mps")  # use "cpu" if MPS/CUDA is unavailable

# Retrieval settings
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
TOP_K_RRF = int(os.getenv("TOP_K_RRF", "10"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# BM25 preprocessing
STOPWORDS = {
    "기준", "여부", "가능", "얼마", "등", "통해", "경우", "사항",
    "어떻게", "무엇", "어디", "언제", "하면", "돼", "해",
}

COMPOUND_TERMS = [
    "어린이보호구역", "어린이 보호구역", "보호구역", "스쿨존",
    "제한속도", "속도제한", "최고속도", "통행속도", "속도위반",
    "주정차위반", "주정차 위반", "주차위반", "정차위반",
    "불법주차", "과태료", "범칙금", "벌점", "부과기준",
    "면허취소", "면허정지", "운전면허", "음주운전", "음주측정",
    "취소처분", "주차단속", "신고", "견인", "조치",
    "휴대폰", "휴대전화", "이동전화", "영상표시장치", "무면허운전",
    "횡단보도", "보행자보호", "일시정지",
]

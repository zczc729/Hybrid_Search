# Hybrid Search 기반 법령 RAG 검색 품질 고도화

> 대한민국 현행 법령 문서를 기반으로 사용자의 법률 질문에 대해 관련 조문을 검색하고, 이를 근거로 답변을 생성하는 법령 특화 RAG 시스템입니다.  
> 기존 Dense Retriever 중심 구조의 한계를 분석하고, BM25 + RRF 기반 Hybrid Search를 적용해 정답 조문 검색 품질을 개선했습니다.

---

## Overview

이 프로젝트는 대한민국 현행 법령 문서를 기반으로 사용자의 질문에 대해 관련 법령 조문을 검색하고, 검색된 조문을 근거로 답변을 생성하는 RAG 시스템입니다.

초기 구조는 Dense Retriever와 Reranker 중심으로 구성했습니다. Dense Search는 의미 기반 검색에 강점이 있지만, 법령 문서에서는 정확한 법령명, 조문 번호, 금액, 기간, 처벌 기준 등 **명시적인 키워드 매칭**이 중요한 경우가 많았습니다.

이를 개선하기 위해 BM25 기반 키워드 검색을 추가하고, Dense Search와 BM25 Search 결과를 RRF 방식으로 결합한 Hybrid Search 구조를 적용했습니다. 이후 Reranker를 통해 최종 후보 문서를 재정렬하고, Local LLM을 통해 근거 기반 답변을 생성하도록 구성했습니다.

---

## Table of Contents

- [Project Goals](#project-goals)
- [Background](#background)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Data Preprocessing](#data-preprocessing)
- [Search System](#search-system)
- [Reranker](#reranker)
- [LLM Answer Generation](#llm-answer-generation)
- [Evaluation](#evaluation)
- [Case Study](#case-study)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [How to Run](#how-to-run)
- [Retrospective](#retrospective)
- [Note](#note)

---

## Project Goals

이 프로젝트의 핵심 목표는 단순히 LLM으로 법률 답변을 생성하는 것이 아니라, **답변의 근거가 되는 법령 조문을 정확히 검색하는 것**입니다.

주요 목표는 다음과 같습니다.

- 대한민국 현행 법령 문서를 조문 단위로 구조화
- Dense Retriever 기반 의미 검색 구현
- BM25 기반 키워드 검색 구현
- RRF 기반 Hybrid Search 구현
- Reranker를 활용한 최종 검색 결과 재정렬
- Local LLM을 활용한 근거 기반 답변 생성
- 검색 결과와 LLM 답변을 분리한 평가 체계 구성

---

## Background

초기 법령 RAG 시스템은 Dense Retriever와 Reranker 중심으로 구성했습니다.

Dense Search는 의미적으로 유사한 조문을 찾는 데 강점이 있지만, 실제 평가 과정에서 다음과 같은 문제가 확인되었습니다.

### 1. 정확한 키워드 기반 질문에 취약

법령 질문에서는 특정 키워드가 정답 조문과 직접 연결되는 경우가 많습니다.

예를 들어 다음과 같은 질문은 의미적 유사도뿐 아니라 법령에 등장하는 핵심 키워드를 정확히 검색하는 것이 중요합니다.

```text
어린이 보호구역에서는 몇 km로 달려야 해?
스쿨존에서 속도위반하면 어떻게 돼?
주정차 위반하면 과태료가 얼마나 나와?
음주운전하면 면허가 바로 취소돼?
```

Dense Search만 사용할 경우 의미적으로 비슷한 문서는 검색되지만, 실제 답변에 필요한 직접 근거 조문이 누락되거나 순위가 낮아지는 문제가 발생했습니다.

### 2. 근거 조문 불일치

일부 질문에서는 답변의 방향은 맞지만, 실제 근거 조문이 잘못 인용되는 문제가 있었습니다.

법령 RAG에서는 답변 내용뿐 아니라 **어떤 조문을 근거로 답변했는지**가 중요합니다.  
따라서 검색 결과의 정확도와 답변의 조문 인용 정확도를 분리해서 평가할 필요가 있었습니다.

### 3. 무관 문서 혼입

질문과 직접 관련성이 낮은 벌칙 조항, 행정 절차 조항, 부칙성 문서가 최종 context에 포함되는 경우도 있었습니다.

이 경우 LLM은 잘못된 context를 기반으로 과잉 생성하거나, 질문과 무관한 내용을 포함한 답변을 생성할 수 있습니다.

---

## Tech Stack

| Category | Stack |
| --- | --- |
| Language | Python |
| PDF Processing | PyMuPDFLoader |
| Embedding Model | jinaai/jina-embeddings-v3 |
| Vector DB | ChromaDB |
| Dense Retriever | Vector Similarity Search |
| Sparse Retriever | BM25 |
| Hybrid Search | Dense Search + BM25 Search + RRF |
| Reranker | multilingual reranker / bge-reranker 계열 |
| LLM | EEVE-Korean-Instruct-10.8B |
| LLM Serving | LM Studio |
| Evaluation | 사용자형 법령 질문 기반 검색 품질 평가 |

---

## Project Structure

```text
.
├── README.md
├── src
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── dense_retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   ├── generator.py
│   └── pipeline.py
├── data_sample
│   ├── sample_questions.csv
│   ├── final_pipeline_summary_sample.csv
│   └── final_pipeline_docs_sample.csv
├── docs
│   ├── evaluation_report.md
│   ├── failure_analysis.md
│   ├── prompt_template.md
│   └── pipeline.md
└── assets
    └── pipeline_diagram.png
```

대용량 원본 PDF와 Vector DB는 저장소에 포함하지 않았습니다.

법령 원문 데이터는 용량이 크고 지속적인 개정 관리가 필요하기 때문에, GitHub에는 샘플 데이터와 평가 결과 일부만 포함하는 방식으로 구성했습니다.

---

## Pipeline

전체 파이프라인은 다음과 같습니다.

```text
User Query
    ↓
Query Preprocessing
    ↓
Dense Search
    ↓
BM25 Search
    ↓
RRF Hybrid Search
    ↓
Reranker
    ↓
Top-k Context Selection
    ↓
Local LLM
    ↓
Final Answer
```

Dense Search는 의미적으로 유사한 조문을 검색하고, BM25 Search는 질문에 포함된 핵심 키워드가 직접 등장하는 조문을 검색합니다.

이후 두 검색 결과를 RRF 방식으로 결합해 후보 문서를 구성하고, Reranker를 통해 질문과 실제 관련성이 높은 문서를 최종 상위에 배치했습니다.

최종적으로 선택된 문서는 Local LLM에 context로 전달되어 법적 근거가 포함된 답변 생성에 사용됩니다.

---

## Data Preprocessing

법령 PDF를 그대로 임베딩할 경우 헤더, 푸터, 페이지 번호, 시행일, 개정 이력 등 검색에 불필요한 정보가 함께 포함됩니다.

이를 줄이기 위해 법령 문서를 조문 단위로 분리하고, 검색에 방해되는 문구를 제거했습니다.

### Preprocessing Steps

- 법제처 / 국가법령정보센터 헤더 제거
- 페이지 번호 및 푸터 제거
- 시행일, 개정 이력 등 검색에 불필요한 문구 제거
- 장, 절, 관 제목 제거
- `제n조`, `제n조의m` 기준 조문 단위 분리
- 법령명, 조문 번호, 원본 파일명 metadata 저장

### Metadata Example

```python
{
    "source": "도로교통법.pdf",
    "law_title": "도로교통법",
    "article": "제12조"
}
```

이를 통해 단순 문단 검색이 아니라, 법령의 구조를 반영한 조문 단위 검색이 가능하도록 구성했습니다.

---

## Search System

검색 시스템은 Dense Search, BM25 Search, RRF 기반 Hybrid Search로 구성했습니다.

### 1. Dense Search

Dense Search는 `jinaai/jina-embeddings-v3`를 사용해 조문을 벡터화하고, ChromaDB에 저장한 뒤 유사도 검색을 수행했습니다.

Dense Search는 질문과 법령 원문의 표현이 다르더라도 의미적으로 유사한 조문을 찾을 수 있다는 장점이 있습니다.

반면 특정 법령명, 조문명, 금액, 처벌 기준처럼 정확한 키워드가 중요한 질문에서는 직접 근거 조문을 놓칠 수 있다는 한계가 있었습니다.

#### Strengths

- 의미 기반 검색에 강함
- 질문 표현과 법령 표현이 달라도 유사한 조문 검색 가능
- 절차형, 개념형, 비교형 질문에서 유용함

#### Limitations

- 정확한 법령명이나 제도명이 중요한 질문에서 정답 조문을 놓칠 수 있음
- 의미적으로 유사하지만 직접 정답이 아닌 조문을 상위에 배치할 수 있음
- 짧거나 구체성이 낮은 질문에서 검색 품질이 흔들릴 수 있음

---

### 2. BM25 Search

BM25 Search는 Dense Search의 한계를 보완하기 위해 추가했습니다.

BM25는 질문에 포함된 핵심 키워드가 문서에 직접 등장하는 경우 강점을 보입니다.  
특히 법령명, 제도명, 처벌 기준, 금액, 기간처럼 정확한 표현이 중요한 질문에서 유용했습니다.

다만 한국어 질문에서는 조사, 어미, 의문어가 검색 품질에 영향을 줄 수 있어 BM25용 질의 전처리를 적용했습니다.

#### Example

사용자 질문:

```text
어린이 보호구역에서는 몇 km로 달려야 해?
```

BM25 검색에 필요한 핵심 토큰:

```text
어린이, 보호구역, 어린이보호구역, km, 속도
```

이처럼 사용자 질문에서 핵심 법령 키워드를 추출해 BM25 검색에 활용했습니다.

#### Strengths

- 정확한 키워드가 포함된 문서 검색에 강함
- 법령명, 제도명, 조문명 검색에 유리함
- 금액, 기간, 처벌 기준 등 명시적 표현 검색에 유리함

#### Limitations

- 사용자 표현과 법령 표현이 다르면 검색 품질이 떨어질 수 있음
- 한국어 조사, 어미, 의문어에 영향을 받을 수 있음
- 의미적으로 유사하지만 키워드가 다른 문서는 찾기 어려움

---

### 3. RRF Hybrid Search

Dense Search와 BM25 Search의 결과를 결합하기 위해 RRF, Reciprocal Rank Fusion 방식을 적용했습니다.

RRF는 각 검색기의 순위를 기반으로 점수를 계산해 여러 검색 결과를 하나의 순위로 통합하는 방식입니다.

#### RRF Score

```text
score(d) = Σ 1 / (k + rank_i(d))
```

- `rank_i(d)` : 각 검색기에서 문서 `d`의 순위
- `k` : 상위 문서에 과도하게 점수가 몰리는 것을 완화하기 위한 상수

#### Purpose

| 목적 | 설명 |
| --- | --- |
| Dense Search 장점 유지 | 의미적으로 유사한 조문을 찾는 능력 유지 |
| BM25 장점 반영 | 정확한 키워드가 포함된 조문을 후보에 포함 |
| 후보 다양성 확보 | 한 검색 방식에만 의존하지 않고 다양한 후보 확보 |
| Recall 개선 | 정답 조문이 최종 후보에 포함될 가능성 증가 |
| Reranker 입력 품질 개선 | 재정렬 단계에 더 적절한 후보 문서 전달 |

---

## Reranker

Hybrid Search로 확보한 후보 문서를 그대로 LLM에 전달하지 않고, Reranker를 통해 질문과 문서의 관련도를 다시 평가했습니다.

검색 단계에서는 정답 후보를 넓게 확보하고, Reranker 단계에서는 최종 context에 들어갈 문서를 더 정밀하게 정렬하도록 구성했습니다.

이를 통해 단순히 검색 점수가 높은 문서가 아니라, 질문에 실제로 더 적합한 조문이 최종 context에 포함되도록 개선했습니다.

### Role of Reranker

- Hybrid Search 결과 재정렬
- 질문과 문서 간 실제 관련성 평가
- 무관한 chunk를 하위로 이동
- 최종 LLM context 품질 개선

---

## LLM Answer Generation

최종 상위 문서는 LM Studio에서 실행한 EEVE-Korean-Instruct 모델에 전달했습니다.

답변 생성 시에는 다음 규칙을 적용했습니다.

- 한국어로 답변
- 단순 질문은 1문장 중심으로 답변
- 복잡한 질문은 2~3문장으로 답변
- 법적 근거는 괄호 안에 표시
- 검색 결과에 없는 내용은 생성하지 않기
- 무관한 조문은 답변에 사용하지 않기

### Answer Example

```text
어린이 보호구역에서는 자동차 등의 통행속도를 시속 30km 이내로 제한할 수 있습니다. (도로교통법 제12조)
```

---

## Evaluation

이번 평가는 도로교통법 관련 사용자형 질문 10개를 기준으로 진행했습니다.

평가에서는 최종 답변의 자연스러움만 확인하지 않고, 검색 결과와 답변 생성을 분리해서 확인했습니다.

### Evaluation Criteria

| 평가 항목 | 설명 |
| --- | --- |
| 정답 조문 포함 여부 | 최종 검색 결과에 직접 근거 조문이 포함되었는지 확인 |
| 최종 순위 | 정답 조문이 최종 몇 위에 배치되었는지 확인 |
| Reranker 정렬 품질 | 관련 문서가 재정렬 후 상위로 올라왔는지 확인 |
| LLM 근거 활용 | 검색된 조문을 기반으로 답변했는지 확인 |
| 조문 번호 정확도 | 답변에 표시된 조문 번호가 실제 근거와 일치하는지 확인 |
| 과잉 생성 여부 | 검색 결과에 없는 내용을 생성했는지 확인 |

### Evaluation Questions

```text
어린이 보호구역에서는 몇 km로 달려야 해?
스쿨존에서 속도위반하면 어떻게 돼?
주정차 위반하면 과태료가 얼마나 나와?
음주운전하면 면허가 바로 취소돼?
불법주차한 차는 신고할 수 있어?
주차위반한 차는 바로 견인될 수 있어?
운전하면서 휴대폰 보면 처벌받아?
음주측정 거부하면 어떻게 돼?
횡단보도에서 보행자는 얼마나 보호받아?
무면허운전하면 처벌이 어떻게 돼?
```

---

## Evaluation Summary

평가 결과, Hybrid Search와 Reranker는 여러 질문에서 직접 근거 조문을 최종 상위 문서에 포함시키는 데 효과가 있었습니다.

특히 어린이 보호구역 제한속도, 횡단보도 보행자 보호, 주차위반 차량 견인 질문에서는 핵심 조문이 최종 상위에 배치되어 검색 품질이 우수했습니다.

반면 일부 질문에서는 검색 결과는 적절했지만 LLM이 조문 번호를 잘못 인용하거나, 검색 단계에서 질문의 핵심 쟁점과 다른 문서가 포함되어 답변 품질이 떨어지는 문제가 있었습니다.

| 구분 | 평가 |
| --- | --- |
| 검색 후보 확보 | 전반적으로 관련 조문을 후보에 포함시키는 데 성공 |
| Reranker 정렬 | 다수 질문에서 핵심 조문을 상위에 배치 |
| LLM 답변 자연성 | 대부분 자연스러운 한국어 답변 생성 |
| 조문 번호 정확도 | 일부 답변에서 잘못된 조문 번호 생성 |
| 과잉 생성 | context가 부정확할 때 무관한 내용 생성 |
| 개선 필요 영역 | 조문 번호 자동 삽입, rule-based filtering, 질문 유형 분류 |

자세한 질문별 검색 결과와 실패 사례 분석은 `docs/evaluation_report.md`에서 확인할 수 있습니다.

---

## Case Study

### Case 1. 성공 사례: 어린이 보호구역 제한속도

#### Question

```text
어린이 보호구역에서는 몇 km로 달려야 해?
```

#### Result

최종 검색 결과에서 `도로교통법 제12조 어린이 보호구역의 지정ㆍ해제 및 관리`가 1위로 검색되었습니다.

해당 조문은 어린이 보호구역에서 자동차 등의 통행속도를 시속 30km 이내로 제한할 수 있다는 내용을 담고 있어 질문에 대한 직접 근거로 적절했습니다.

#### Analysis

이 사례는 Hybrid Search와 Reranker가 정답 조문을 안정적으로 찾아낸 대표 사례입니다.

Dense Search는 의미 기반 후보를 확보했고, BM25는 `어린이 보호구역`, `속도`, `30km`와 같은 키워드 기반 후보를 보완했습니다. 이후 Reranker를 통해 핵심 조문이 최종 상위에 유지되었습니다.

---

### Case 2. 성공 사례: 주차위반 차량 견인

#### Question

```text
주차위반한 차는 바로 견인될 수 있어?
```

#### Result

최종 검색 결과에서 `도로교통법 시행령 제13조 주차위반 차의 견인ㆍ보관 및 반환 등을 위한 조치`와 `도로교통법 시행규칙 제22조 주차위반차의 견인ㆍ보관 및 반환 등을 위한 조치 등`이 상위에 배치되었습니다.

해당 문서들은 주차위반 차량의 견인 절차와 견인 전 필요한 조치에 관한 내용을 포함하고 있어 질문과 직접적으로 관련이 있었습니다.

#### Analysis

이 사례는 단순 처벌 여부가 아니라 행정 절차를 묻는 질문에서도 검색 결과가 적절하게 구성된 사례입니다.

다만 최종 답변에서 조문 번호를 부정확하게 언급한 부분이 있어, 검색 품질은 우수하지만 답변 생성 단계의 조문 인용 방식은 개선이 필요하다고 판단했습니다.

---

### Case 3. 개선 필요 사례: 운전 중 휴대폰 사용

#### Question

```text
운전하면서 휴대폰 보면 처벌받아?
```

#### Result

운전 중 휴대폰 사용과 직접 관련된 조문이 최종 상위에 충분히 포함되지 않았습니다.

그 결과 LLM 답변에서는 음주운전, 음주측정, 경찰관 요구 불응 등 질문과 무관한 내용이 섞이는 문제가 발생했습니다.

#### Cause

사용자 표현인 `휴대폰`과 법령 표현인 `이동전화`, `영상표시장치` 등이 제대로 매핑되지 않아 직접 근거 조문 검색에 실패한 것으로 분석했습니다.

#### Improvement

- 법령 도메인 동의어 사전 구축
- `휴대폰`, `휴대전화`, `이동전화`, `영상표시장치` 등 표현 매핑
- 질문 핵심 키워드가 포함되지 않은 문서에 대한 rule-based filtering 적용
- 검색 결과에 직접 근거가 없을 경우 LLM이 단정하지 않도록 프롬프트 개선

---

### Case 4. 개선 필요 사례: 스쿨존 속도위반 처벌

#### Question

```text
스쿨존에서 속도위반하면 어떻게 돼?
```

#### Result

검색 결과에는 어린이 보호구역 제한속도와 관련된 `도로교통법 제12조`가 포함되었습니다.

하지만 사용자의 질문은 단순히 제한속도를 묻는 것이 아니라, 속도위반 시 과태료, 범칙금, 벌점 등 위반 결과를 묻는 질문이었습니다.

최종 context에는 위반 결과와 관련된 문서가 충분히 포함되지 않았고, LLM 답변에서도 검색 context와 맞지 않는 조문 번호가 생성되었습니다.

#### Cause

질문 유형이 `규정 확인형`인지 `위반 처분형`인지 구분되지 않아, 제한속도 조문은 검색되었지만 처분 기준 문서가 충분히 검색되지 않았습니다.

#### Improvement

- 질문 유형을 규정 확인형, 위반 처분형, 신고 절차형 등으로 분류
- `위반하면`, `처벌`, `과태료`, `범칙금`, `벌점` 키워드가 포함될 경우 관련 문서 가중치 부여
- LLM이 조문 번호를 직접 생성하지 않고 metadata 기반으로 인용하도록 개선

---

## Limitations

| 한계 | 설명 |
| --- | --- |
| 사용자 표현과 법령 표현 차이 | `휴대폰`과 `이동전화`처럼 표현 차이가 있는 경우 검색 품질 저하 |
| 질문 유형 구분 부족 | 규정 확인형, 위반 처분형, 신고 절차형 질문을 동일한 방식으로 처리 |
| 무관 문서 혼입 | 일부 질문에서 직접 관련성이 낮은 문서가 최종 context에 포함 |
| 조문 번호 오인용 | LLM이 검색 결과와 다른 조문 번호를 생성하는 경우 발생 |
| 긴 조문 처리 | 하나의 조문 안에 여러 항목이 있는 경우 세부 조건 검색이 어려움 |
| 비법령 데이터 부족 | 신고 절차, 행정해석, 고시 등 실무 정보 반영에 한계 |

---

## Future Work

- 법령 도메인 동의어 사전 구축
- 질문 유형 분류기 추가
- Adaptive Hybrid Search 적용
- Reranker 이후 rule-based filtering 추가
- 조문 번호를 LLM이 생성하지 않고 metadata에서 자동 삽입하도록 개선
- 항·호 단위 chunking 적용
- Recall@K, MRR, Precision@K 기반 정량 평가 추가
- 판례, 행정해석, 고시, 보도자료 등 비법령 문서 추가
- 법령 개정 이력 및 최신성 관리 구조 추가

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run LM Studio

LM Studio에서 EEVE-Korean-Instruct-10.8B 모델을 로드한 뒤 Local Server를 실행합니다.

예시:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
```

### 3. Preprocess Documents

```bash
python src/preprocessing.py
```

### 4. Build Vector DB

```bash
python src/embedding.py
```

### 5. Run Pipeline

```bash
python src/pipeline.py
```

---

## Retrospective

이번 프로젝트는 기존 Dense Retriever 중심 법령 RAG 시스템의 한계를 분석하고, 이를 개선하기 위해 BM25와 RRF 기반 Hybrid Search를 적용한 검색 품질 고도화 프로젝트입니다.

평가 결과, Hybrid Search와 Reranker는 어린이 보호구역 제한속도, 횡단보도 보행자 보호, 주차위반 차량 견인처럼 핵심 근거 조문이 명확한 질문에서 좋은 성능을 보였습니다.

Dense Search는 의미 기반 후보 확보에 강점이 있었고, BM25는 정확한 키워드 기반 후보 확보에 강점이 있었습니다. RRF를 통해 두 검색 결과를 결합함으로써 한쪽 검색 방식만 사용할 때보다 더 다양한 후보 문서를 확보할 수 있었습니다.

반면 운전 중 휴대폰 사용, 스쿨존 속도위반 처벌처럼 사용자 표현과 법령 표현이 다르거나, 단순 규정이 아닌 처분 기준을 묻는 질문에서는 여전히 한계가 있었습니다.

또한 검색 결과가 적절하더라도 LLM이 조문 번호를 잘못 인용하는 사례가 있어, 최종 답변의 신뢰성을 높이기 위해서는 metadata 기반 조문 번호 삽입, rule-based filtering, 답변 후처리 과정이 필요하다는 점을 확인했습니다.

이 프로젝트를 통해 RAG 시스템의 품질은 LLM 성능만으로 결정되지 않으며, 검색기 설계, 문서 전처리, 후보 문서 재정렬, 프롬프트 설계, 후처리까지 전체 파이프라인이 함께 작동해야 한다는 점을 경험했습니다.

---

## Note

본 프로젝트는 학습 및 포트폴리오 목적으로 진행한 개인 프로젝트입니다.  
법률 답변은 실제 법률 자문이 아니며, 법적 판단이 필요한 경우 전문가의 검토가 필요합니다.

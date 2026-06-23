# Security Review

GitHub 공개용 정리 과정에서 다음 항목을 확인했습니다.

- 노트북 출력 제거
- 로컬 사용자 경로 마스킹
- API Key, Token, Password 등 민감정보 미포함 확인
- 원본 PDF, Vector DB, 모델 파일 등 대용량 파일 제외
- LM Studio 기본 API Key는 로컬 테스트용 문자열인 `lm-studio`만 사용
- 실행 경로는 `data/raw`, `chroma_store` 등 상대 경로로 정리

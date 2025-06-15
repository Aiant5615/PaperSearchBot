# PaperSearchBot

논문 검색 및 질의응답 시스템입니다. 이 프로젝트는 arXiv 논문을 크롤링하고, RAG(Retrieval-Augmented Generation) 기술을 활용하여 논문에 대한 질문에 답변할 수 있는 시스템을 제공합니다.

## 주요 기능

- arXiv 논문 크롤링 및 데이터 추출
- 논문 내용의 임베딩 생성 및 저장
- 자연어 기반 논문 필터링
- 시맨틱 검색을 통한 관련 논문 검색
- 논문 내용에 대한 질의응답

## 기술 스택

- Python 3.x
- PostgreSQL (pgvector 확장 사용)
- OpenAI API
- Streamlit
- BeautifulSoup4
- scikit-learn

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd PaperSearchBot
```

2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
`.env` 파일을 생성하고 다음 변수들을 설정합니다:
```
OPENAI_API_KEY=your_openai_api_key
DBHOST=your_db_host
DBPORT=5432
DBNAME=your_db_name
DBUSER=your_db_user
DBPASS=your_db_password
```

## 데이터 준비

1. arXiv 데이터셋 다운로드
   - [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)에서 데이터셋을 다운로드합니다.
   - Kaggle 계정이 필요하며, 데이터셋 사용을 위해 Kaggle API 토큰을 설정해야 할 수 있습니다.

2. 데이터 압축 해제
   - 다운로드한 데이터셋을 `data` 폴더에 압축 해제합니다.
   - 데이터셋은 JSON 형식으로 제공되며, 메타데이터와 논문 내용을 포함합니다.

## 데이터베이스 설정

1. PostgreSQL 설치 및 pgvector 확장 활성화
2. 다음 스키마를 생성:
   - paper_schema.paper
   - paper_schema.author
   - paper_schema.paper_author
   - paper_schema.category
   - paper_schema.paper_category

## 실행 방법

1. 데이터 수집 및 처리
```bash
python rag.py
```

2. 웹 인터페이스 실행
```bash
streamlit run query_data.py
```

## 사용 방법

1. 웹 인터페이스에서 필터 조건 입력 (예: "Papers by Alice Zhang since 2021")
2. 검색 쿼리 입력
3. "Run Search" 버튼 클릭
4. 검색 결과 확인 및 특정 논문에 대한 질문 입력

## 주의사항

- OpenAI API 키가 필요합니다
- PostgreSQL 데이터베이스가 필요합니다
- 충분한 디스크 공간이 필요합니다 (논문 데이터 저장용)
- Kaggle 계정과 API 토큰이 필요할 수 있습니다

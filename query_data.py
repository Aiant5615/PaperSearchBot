import os
from datetime import date
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError
from pgvector.psycopg2 import register_vector
from Embeddings import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import streamlit as st

# ─── 1) 환경변수 및 DB 연결 ──────────────────────────────────────
load_dotenv(override=True)
try:
    conn = psycopg2.connect(
        host=os.getenv("DBHOST"),
        port=os.getenv("DBPORT", "5432"),
        dbname=os.getenv("DBNAME"),
        user=os.getenv("DBUSER"),
        password=os.getenv("DBPASS"),
        sslmode="disable",
    )
    register_vector(conn)
except OperationalError as e:
    st.error(f"DB 연결 실패: {e}")
    st.stop()

cur = conn.cursor()

# ─── 2) 프롬프트 템플릿 정의 (제목 기반 출처 표시) ─────────────────────
PROMPT_TEMPLATE = """
다음 논문 정보만을 참고하여 질문에 답하세요:

{context}

---

질문: {question}

출처는 논문의 제목을 대괄호 [ ] 안에 표시해주세요.
"""


# ─── 3) 검색 함수 정의 ───────────────────────────────────────────
def search_after_date(query: str, start_date: str, k: int = 3):
    embed_query = get_embedding_function().embed_query(query)
    vec_literal = "[" + ",".join(str(x) for x in embed_query) + "]"
    sql = """
    SELECT paper_id, title, abstract, update_date
      FROM paper_schema.paper
     WHERE update_date >= %s::date
     ORDER BY embedding <=> %s::vector
     LIMIT %s;
    """
    cur.execute(sql, (start_date, vec_literal, k))
    return cur.fetchall()


# ─── 4) 세션 상태 초기화 ─────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today().isoformat()
if "results" not in st.session_state:
    st.session_state.results = []

# ─── 5) Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="Date-Filtered Semantic Search")
st.title("📑 날짜 이후 논문 유사도 검색")

# 입력 위젯
st.session_state.query = st.text_input(
    "🔍 검색할 키워드를 입력하세요", value=st.session_state.query
)
date_input = st.date_input(
    "📅 필터 시작 날짜 선택",
    value=date.fromisoformat(st.session_state.start_date),
    min_value=date(1900, 1, 1),
)

# ─── 6) 검색 실행 버튼 ───────────────────────────────────────────
if st.button("검색 실행"):
    if not st.session_state.query:
        st.warning("키워드를 입력해 주세요.")
    else:
        st.session_state.start_date = date_input.isoformat()
        with st.spinner("검색 중…"):
            st.session_state.results = search_after_date(
                st.session_state.query, st.session_state.start_date, k=3
            )

# ─── 7) 검색 결과 표시 ───────────────────────────────────────────
if st.session_state.results:
    st.success(
        f"{st.session_state.start_date} 이후 업데이트된 논문 중 "
        f"'{st.session_state.query}'와 유사도 상위 {len(st.session_state.results)}건:"
    )
    for idx, (pid, title, abstract, upd) in enumerate(st.session_state.results, 1):
        st.markdown(
            f"**{idx}. {title}**  \n"
            f"- ID: `{pid}`  \n"
            f"- update_date: {upd}  \n\n"
            f"{abstract[:200].replace(chr(10),' ')}…"
        )
        st.write("---")

    # ─── 8) LLM 답변 생성 버튼 ───────────────────────────────────
    if st.button("LLM에게 답변 생성"):
        context = "\n\n---\n\n".join(
            f"제목: {t}\n초록: {a}" for _, t, a, _ in st.session_state.results
        )
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context, question=st.session_state.query
        )
        model = Ollama(model="mistral")
        try:
            with st.spinner("LLM 응답 생성 중…"):
                answer = model.invoke(prompt)
            st.write(answer)
        except Exception as e:
            st.error(f"모델 호출 중 오류 발생: {e}")

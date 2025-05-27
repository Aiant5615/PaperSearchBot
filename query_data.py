import os
import traceback
import concurrent.futures
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
    cur = conn.cursor()
except Exception as e:
    st.error("❌ DB 연결 실패")
    st.error(str(e))
    st.error(traceback.format_exc())
    st.stop()


# ─── 스키마 정보 조회 함수 ───────────────────────────────────────
def show_db_schema(cur):
    try:
        cur.execute(
            """
            SELECT tablename
              FROM pg_catalog.pg_tables
             WHERE schemaname = 'paper_schema';
        """
        )
        tables = [row[0] for row in cur.fetchall()]
        st.info(f"🗃️ 사용 가능한 테이블: {', '.join(tables)}")
        for tbl in tables:
            cur.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = 'paper_schema'
                   AND table_name = %s;
            """,
                (tbl,),
            )
            cols = [r[0] for r in cur.fetchall()]
            st.info(f"• `{tbl}` 컬럼: {', '.join(cols)}")
    except Exception as e:
        st.error(f"⚠️ 스키마 정보 조회 실패: {e}")


# ─── 2) 세션 상태 초기화 ─────────────────────────────────────────
for key, default in [
    ("where_clause", ""),
    ("total_count", 0),
    ("results", []),
    ("answer", None),
    ("filter_nl", ""),
    ("query_nl", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── 3) Few-shot prompt 정의 ─────────────────────────────────────
FILTER_PROMPT = """
Below is the database schema.

Tables:
- paper_schema.paper(paper_id, submitter, title, abstract, update_date, embedding)
- paper_schema.author(author_id, last_name, first_name)
- paper_schema.paper_author(paper_id, author_id, author_order)
- paper_schema.category(category_id, code, description)
- paper_schema.paper_category(paper_id, category_id)

**When there are multiple conditions, join them with `AND`.**  
Generate **only** a single-line, valid SQL WHERE clause (omit the “WHERE” keyword entirely).  
Do **NOT** include line breaks, comments, or explanations—just one SQL expression.

### Example 1: Author only  
Natural language: Papers published by Alice Zhang  
WHERE clause: EXISTS(SELECT 1 FROM paper_schema.paper_author pa JOIN paper_schema.author a ON pa.author_id=a.author_id WHERE pa.paper_id=paper_schema.paper.paper_id AND a.first_name='Alice' AND a.last_name='Zhang')

### Example 2: Submitter only  
Natural language: Papers submitted by Alice Zhang  
WHERE clause: submitter = 'Alice Zhang'

### Example 3: Date only  
Natural language: Papers updated after 2019-06  
WHERE clause: update_date >= '2019-06-01'

### Example 4: Author + Date  
Natural language: Papers published by Alice Zhang since 2021-05  
WHERE clause: update_date >= '2021-05-01' AND EXISTS(SELECT 1 FROM paper_schema.paper_author pa JOIN paper_schema.author a ON pa.author_id=a.author_id WHERE pa.paper_id=paper_schema.paper.paper_id AND a.first_name='Alice' AND a.last_name='Zhang')

### Example 5: Category + Date  
Natural language: Papers in category 'cs.AI' after 2020-07  
WHERE clause: update_date >= '2020-07-01' AND EXISTS(SELECT 1 FROM paper_schema.paper_category pc JOIN paper_schema.category c ON pc.category_id=c.category_id WHERE pc.paper_id=paper_schema.paper.paper_id AND c.code='cs.AI')

### Example 6: Author + Category + Date  
Natural language: Papers by Bob Lee in category 'hep-th' published since 2019-03  
WHERE clause: update_date >= '2019-03-01' AND EXISTS(SELECT 1 FROM paper_schema.paper_author pa JOIN paper_schema.author a ON pa.author_id=a.author_id WHERE pa.paper_id=paper_schema.paper.paper_id AND a.first_name='Bob' AND a.last_name='Lee') AND EXISTS(SELECT 1 FROM paper_schema.paper_category pc JOIN paper_schema.category c ON pc.category_id=c.category_id WHERE pc.paper_id=paper_schema.paper.paper_id AND c.code='hep-th')

### Your turn  
Natural language: {nl_filter}  
WHERE clause:
"""


def get_where_clause(nl_filter: str, timeout_sec: int = 15) -> str:
    prompt = FILTER_PROMPT.format(nl_filter=nl_filter)
    model = Ollama(model="llama3.2:3b")
    try:
        with concurrent.futures.ThreadPoolExecutor() as exe:
            future = exe.submit(model.invoke, prompt)
            raw = future.result(timeout=timeout_sec).strip()
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"LLM invocation timed out after {timeout_sec} seconds")
    except Exception as e:
        raise RuntimeError(f"Error during WHERE-generation: {e}")
    if raw.upper().startswith("WHERE "):
        raw = raw[6:]
    return raw


# ─── 4) 검색 결과로 답변 생성용 프롬프트 ───────────────────────
ANSWER_PROMPT = """
Using only the information from the following papers, answer the question.

Papers:
{context}

Question: {question}

Please cite each paper by enclosing its title in square brackets, e.g. [Title].
"""

# ─── 5) Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="📑 NL Filter + Semantic Search")
st.title("🔎 Search my paper")

st.session_state.filter_nl = st.text_input(
    "Enter your filter condition:", value=st.session_state.filter_nl
)
st.session_state.query_nl = st.text_input(
    "Enter your research query:", value=st.session_state.query_nl
)

if st.button("Run Search", key="run_search"):
    if not st.session_state.filter_nl or not st.session_state.query_nl:
        st.warning("Please provide both a filter condition and a search query.")
    else:
        # 1) WHERE 절 생성
        try:
            with st.spinner("Generating SQL filter..."):
                st.session_state.where_clause = get_where_clause(
                    st.session_state.filter_nl, timeout_sec=15
                )
        except Exception as e:
            st.error(f"❌ WHERE-gen error: {e}")
            show_db_schema(cur)
            st.stop()

        # 2) 필터된 논문 개수 조회
        try:
            count_sql = f"SELECT COUNT(*) FROM paper_schema.paper WHERE {st.session_state.where_clause};"
            cur.execute(count_sql)
            st.session_state.total_count = cur.fetchone()[0]
        except Exception as e:
            st.error("❌ COUNT 쿼리 오류")
            st.error("SQL: " + count_sql)
            st.error(str(e))
            show_db_schema(cur)
            st.stop()

        # 필터 결과가 0건일 때 예시 submitter 제공
        if st.session_state.total_count == 0:
            st.warning("⚠️ 필터에 맞는 논문이 없습니다. 조건을 다시 확인해 보세요.")
            st.stop()

        # 3) 유사도 상위 3개 논문 조회
        try:
            embed_query = get_embedding_function().embed_query(
                st.session_state.query_nl
            )
            vec_literal = "[" + ",".join(str(x) for x in embed_query) + "]"
            k = 3
            sim_sql = f"""
                SELECT paper_id, title, abstract, update_date
                  FROM paper_schema.paper
                 WHERE {st.session_state.where_clause}
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s;
            """
            with st.spinner("Searching top-k papers..."):
                cur.execute(sim_sql, (vec_literal, k))
                st.session_state.results = cur.fetchall()
        except Exception as e:
            st.error("❌ 유사도 검색 오류")
            st.error("SQL: " + sim_sql)
            st.error("Params: " + repr((vec_literal, k)))
            st.error(str(e))
            show_db_schema(cur)
            st.stop()

        st.session_state.answer = None

# ─── 6) 결과 및 답변 생성 UI ───────────────────────────────────
if st.session_state.results:
    st.code(
        f"/* WHERE clause */\nWHERE {st.session_state.where_clause}", language="sql"
    )
    st.info(f"Filtered papers count: **{st.session_state.total_count}**")
    st.success(f"Top **{len(st.session_state.results)}** similar papers:")

    for i, (pid, title, abstract, upd) in enumerate(st.session_state.results, 1):
        st.markdown(f"**{i}. {title}**  \n- ID: `{pid}`, Updated: {upd}")
        st.write(abstract[:200].replace("\n", " ") + "…")
        st.write("---")

    if st.button("Generate Answer", key="gen_answer"):
        try:
            context = "\n\n---\n\n".join(
                f"Title: {t}\nAbstract: {a}" for _, t, a, _ in st.session_state.results
            )
            prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT).format(
                context=context, question=st.session_state.query_nl
            )
            st.session_state.answer = Ollama(model="llama3.2:3b").invoke(prompt)
        except Exception as e:
            st.error("❌ Answer-generation error")
            st.error(str(e))
            show_db_schema(cur)

if st.session_state.answer:
    st.subheader("💬 Generated Answer")
    st.write(st.session_state.answer)

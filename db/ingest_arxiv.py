import psycopg2
import json
from dateutil import parser
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
import numpy as np
from datetime import date


def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings


file_path = "data/arxiv-metadata-oai-snapshot.json"

# 0) 전체 줄 수 계산 (한 번만 빠르게 순회)
with open(file_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# 1) PostgreSQL 연결 설정
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="dellaanima",
    host="localhost",
    port="5432",
)
cur = conn.cursor()


# 2) 실제 삽입/업데이트 루프
with open(file_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(
        tqdm(f, total=total_lines, desc="논문 삽입", unit="줄"), start=1
    ):
        data = json.loads(line)
        ### Temp : Data 너무 많아서 우선 2024-01-01 이후것만 ####
        # update_date 파싱
        update_date = parser.parse(data.get("update_date")).date()

        # ─── 2024년 1월 1일 이전이면 스킵 ─────────────────────────────
        if update_date < date(2024, 1, 1):
            continue
        ### Temp : Data 너무 많아서 우선 2024-01-01 이후것만 ####


        paper_id = data["id"]
        submitter = data.get("submitter")
        title = data.get("title")
        comments = data.get("comments")
        journal_ref = data.get("journal-ref")
        doi = data.get("doi")
        report_no = data.get("report-no")
        abstract = data.get("abstract")
        update_date = parser.parse(data.get("update_date")).date()

        # ─── paper 테이블: ON CONFLICT (doi) DO UPDATE ─────────────────
        if doi:
            sql = """
            INSERT INTO paper_schema.paper
                (paper_id, submitter, title, comments, journal_ref,
                 doi, report_no, abstract, update_date)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (doi) DO UPDATE
                SET submitter   = EXCLUDED.submitter,
                    title       = EXCLUDED.title,
                    comments    = EXCLUDED.comments,
                    journal_ref = EXCLUDED.journal_ref,
                    report_no   = EXCLUDED.report_no,
                    abstract    = EXCLUDED.abstract,
                    update_date = EXCLUDED.update_date
            WHERE paper_schema.paper.update_date < EXCLUDED.update_date
            RETURNING paper_id;
            """
            cur.execute(
                sql,
                (
                    paper_id,
                    submitter,
                    title,
                    comments,
                    journal_ref,
                    doi,
                    report_no,
                    abstract,
                    update_date,
                ),
            )
            row = cur.fetchone()
            if row:
                record_paper_id = row[0]
            else:
                cur.execute(
                    "SELECT paper_id FROM paper_schema.paper WHERE doi = %s",
                    (doi,),
                )
                record_paper_id = cur.fetchone()[0]

        # ─── doi 없는 경우: paper_id 기준 INSERT ────────────────────────
        else:
            sql = """
            INSERT INTO paper_schema.paper
                (paper_id, submitter, title, comments, journal_ref,
                 doi, report_no, abstract, update_date)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (paper_id) DO NOTHING
            RETURNING paper_id;
            """
            cur.execute(
                sql,
                (
                    paper_id,
                    submitter,
                    title,
                    comments,
                    journal_ref,
                    doi,
                    report_no,
                    abstract,
                    update_date,
                ),
            )
            row = cur.fetchone()
            record_paper_id = row[0] if row else paper_id

        # ─── embedding 칼럼 업데이트 (기존 값이 NULL일 때만) ─────────────────
        cur.execute(
            "SELECT embedding IS NOT NULL FROM paper_schema.paper WHERE paper_id = %s",
            (record_paper_id,),
        )
        has_vec = cur.fetchone()[0]
        if not has_vec:
            vec = get_embedding_function().embed_query(abstract)
            cur.execute(
                """
                UPDATE paper_schema.paper
                   SET embedding = %s
                 WHERE paper_id = %s
                """,
                (vec, record_paper_id),
            )

        # ─── version 테이블 업데이트 ───────────────────────────────────
        for v in data.get("versions", []):
            version = v.get("version")
            created_at = parser.parse(v.get("created"))
            cur.execute(
                """
                INSERT INTO paper_schema.version
                    (paper_id, version, created_at)
                VALUES (%s,%s,%s)
                ON CONFLICT DO NOTHING
                """,
                (record_paper_id, version, created_at),
            )

        # ─── author & paper_author 테이블 업데이트 ───────────────────────
        for order, names in enumerate(data.get("authors_parsed", []) or [], start=1):
            padded = names + [None, None]
            last, first, middle = padded[0], padded[1], padded[2]

            cur.execute(
                """
                SELECT author_id FROM paper_schema.author
                 WHERE last_name = %s AND first_name = %s
                   AND (middle_name = %s OR (middle_name IS NULL AND %s IS NULL))
                """,
                (last, first, middle, middle),
            )
            row = cur.fetchone()
            if row:
                author_id = row[0]
            else:
                cur.execute(
                    """
                    INSERT INTO paper_schema.author
                      (last_name, first_name, middle_name)
                    VALUES (%s,%s,%s)
                    RETURNING author_id
                    """,
                    (last, first, middle),
                )
                author_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO paper_schema.paper_author
                  (paper_id, author_id, author_order)
                VALUES (%s,%s,%s)
                ON CONFLICT DO NOTHING
                """,
                (record_paper_id, author_id, order),
            )

        # ─── category & paper_category 테이블 업데이트 ───────────────────
        for code in data.get("categories", "").split():
            cur.execute(
                "SELECT category_id FROM paper_schema.category WHERE code = %s",
                (code,),
            )
            row = cur.fetchone()
            if row:
                category_id = row[0]
            else:
                cur.execute(
                    """
                    INSERT INTO paper_schema.category
                      (code, description)
                    VALUES (%s, %s)
                    RETURNING category_id
                    """,
                    (code, None),
                )
                category_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO paper_schema.paper_category
                  (paper_id, category_id)
                VALUES (%s,%s)
                ON CONFLICT DO NOTHING
                """,
                (record_paper_id, category_id),
            )

        # 100줄 단위로 중간 커밋
        if idx % 100 == 0:
            conn.commit()

# 3) 최종 커밋 및 종료
conn.commit()
cur.close()
conn.close()
print("✅ JSON → DB 삽입 완료")

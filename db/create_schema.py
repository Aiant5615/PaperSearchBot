import psycopg2

# DB 접속 정보
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",  # mac 사용자 이름
    password="dellaanima",  # password 없는 경우는 비워둬도 됨 (trust 인증일 경우)
    host="localhost",
    port="5432",
)

# 커서 생성
cur = conn.cursor()

# 스키마 및 테이블 생성 SQL
schema_sql = """
CREATE SCHEMA IF NOT EXISTS paper_schema;

CREATE TABLE IF NOT EXISTS paper_schema.paper (
    paper_id TEXT PRIMARY KEY,
    submitter TEXT NOT NULL,
    title TEXT NOT NULL,
    comments TEXT,
    journal_ref TEXT,
    doi TEXT UNIQUE,
    report_no TEXT,
    abstract TEXT,
    update_date DATE,
    embedding vector(768)
);

CREATE TABLE IF NOT EXISTS paper_schema.version (
    version_id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES paper_schema.paper(paper_id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_schema.author (
    author_id SERIAL PRIMARY KEY,
    last_name TEXT NOT NULL,
    first_name TEXT NOT NULL,
    middle_name TEXT
);

CREATE TABLE IF NOT EXISTS paper_schema.paper_author (
    paper_id TEXT REFERENCES paper_schema.paper(paper_id) ON DELETE CASCADE,
    author_id INTEGER REFERENCES paper_schema.author(author_id) ON DELETE CASCADE,
    author_order INTEGER NOT NULL,
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE IF NOT EXISTS paper_schema.category (
    category_id SERIAL PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS paper_schema.paper_category (
    paper_id TEXT REFERENCES paper_schema.paper(paper_id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES paper_schema.category(category_id) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, category_id)
);
"""

# SQL 실행 및 커밋
cur.execute(schema_sql)
conn.commit()

# 연결 종료
cur.close()
conn.close()

print("✅ paper_schema 및 모든 테이블 생성 완료")

#!/usr/bin/env python
# enable_pgvector.py

import os
import psycopg2
from psycopg2 import sql, OperationalError
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────────────────────────────
# 1) .env에서 DB 접속 정보 로드
# ───────────────────────────────────────────────────────────────────────────────
load_dotenv(override=True)
DBUSER = os.getenv("DBUSER", "postgres")  # 기본값 postgres
DBPASS = os.getenv("DBPASS", "")
DBHOST = os.getenv("DBHOST", "localhost")
DBPORT = os.getenv("DBPORT", "5432")
DBNAME = os.getenv("DBNAME", "postgres")

# ───────────────────────────────────────────────────────────────────────────────
# 2) PostgreSQL에 연결하고 pgvector 확장 설치
# ───────────────────────────────────────────────────────────────────────────────
try:
    conn = psycopg2.connect(
        host=DBHOST,
        port=DBPORT,
        dbname=DBNAME,
        user=DBUSER,
        password=DBPASS,
        sslmode="disable" if DBHOST in ("localhost", "127.0.0.1") else "require",
    )
    conn.autocommit = True  # CREATE EXTENSION은 트랜잭션 블록 밖에서 실행해야 함

    with conn.cursor() as cur:
        # vector 확장 설치 여부 확인 후 없으면 생성
        cur.execute(
            """
            SELECT 1
              FROM pg_extension
             WHERE extname = 'vector';
        """
        )
        exists = cur.fetchone()

        if exists:
            print("✅ pgvector extension already installed.")
        else:
            print("🔧 Installing pgvector extension...")
            cur.execute(sql.SQL("CREATE EXTENSION vector;"))
            print("✅ pgvector extension has been installed.")

except OperationalError as e:
    print(f"❌ 데이터베이스 연결 실패: {e}")
finally:
    if conn:
        conn.close()

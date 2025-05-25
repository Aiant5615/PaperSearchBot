import os
import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Embeddings import get_embedding_function

# set up pg vector DB
load_dotenv(override=True)
DBUSER = os.environ["DBUSER"]
DBPASS = os.environ["DBPASS"]
DBHOST = os.environ["DBHOST"]
DBNAME = os.environ["DBNAME"]
DATABASE_URI = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}"

# Use SSL if not connecting to localhost
DBSSL = "disable"
if DBHOST != "localhost":
    DBSSL = "require"

# Connect to PostgreSQL database in Timescale using connection string
conn = psycopg2.connect(
    host=DBHOST,                   
    database=DBNAME,           
    user=DBUSER,                   
    password=DBPASS,             
    port=5432                     
)
cur = conn.cursor()

# install pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()    

# Register the vector type with psycopg2
register_vector(conn)

def create_table():
    # Create table to store embeddings and metadata
    table_create_command = """
    DROP TABLE IF EXISTS embeddings;
    CREATE TABLE embeddings (
                id serial primary key, 
                source text,
                page text,
                content text,
                chunk_id integer,
                embedding vector(768)
                );
                """
    cur.execute(table_create_command)
    conn.commit()
    print("Table created!")

def document_loader():
    loader = PyPDFLoader("data/Beginners_Guide_Baseball.pdf")
    pages = loader.load_and_split()
    return(pages)

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,)
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    documents = document_loader()
    chunks = split_documents(documents)
    create_table()
    cur.execute("CREATE INDEX ON embeddings USING hnsw (embedding vector_l2_ops)")
    emb_chunks = get_embedding_function().embed_documents([chunk.page_content for chunk in chunks])
    for i, v in enumerate(emb_chunks):
        source = chunks[i].metadata['source']
        page = chunks[i].metadata['page']
        content = chunks[i].page_content
        cur.execute("INSERT INTO embeddings (source, page, content, chunk_id,embedding) VALUES (%s, %s, %s, %s, %s)", (source, page, content, i, v,))
        conn.commit()

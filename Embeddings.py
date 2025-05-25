from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text", )
    return embeddings

# if __name__ == "__main__":
#     vector = get_embedding_function().embed_query('Testing the embedding model')
#     print(len(vector))
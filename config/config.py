import os
from dotenv import load_dotenv

load_dotenv()

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Retrieval
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 5))
TOP_K_SEMANTIC = int(os.getenv("TOP_K_SEMANTIC", 5))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 3))

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Vector store
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "healthcare_docs")

# Paths
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "./data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "./data/processed")

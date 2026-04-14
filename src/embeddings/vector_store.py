from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)


class VectorStore:
    """Wraps ChromaDB and a SentenceTransformer embedding model."""

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' ready at: {CHROMA_PERSIST_DIR}")

    def embed_and_store(self, chunks: List[Document]) -> None:
        """Embed a list of Document chunks and upsert them into ChromaDB."""
        if not chunks:
            print("No chunks to store.")
            return

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{chunk.metadata.get('chunk_id', i)}" for i, chunk in enumerate(chunks)]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_model.encode(texts, show_progress_bar=True).tolist()

        # Upsert in batches of 500 to avoid memory issues
        batch_size = 500
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            self.collection.upsert(
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
            print(f"  Stored batch {start // batch_size + 1}: chunks {start}–{min(end, len(texts))}")

        print(f"\nDone. Total vectors in store: {self.collection.count()}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Return the top-k most similar chunks for a query."""
        query_embedding = self.embed_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    # Quick test: ingest the Executive Summary PDF
    from src.ingestion.document_loader import load_all_pdfs

    chunks = load_all_pdfs()
    store = VectorStore()
    store.embed_and_store(chunks)

    # Test a query
    print("\nTest query: 'first-line treatment for hypertension in diabetes'")
    results = store.similarity_search("first-line treatment for hypertension in diabetes", k=3)
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Source: {doc.metadata.get('file_name')} | Page: {doc.metadata.get('page')}")
        print(f"     {doc.page_content[:200]}")

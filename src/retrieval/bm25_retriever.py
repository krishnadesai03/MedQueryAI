"""
BM25 keyword retriever backed by rank_bm25.

The index is built from chunks already stored in ChromaDB so there is
a single source of truth — no separate document store needed.
The serialised index is saved to data/processed/bm25_index.pkl so it
only needs to be rebuilt when new data is ingested.
"""
import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from langchain.schema import Document

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import PROCESSED_DATA_DIR

INDEX_PATH = Path(PROCESSED_DATA_DIR) / "bm25_index.pkl"


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    """Thin wrapper around BM25Okapi that stores corpus documents for retrieval."""

    def __init__(self, documents: List[Document] | None = None):
        """
        Pass documents to build a fresh index, or omit to load from disk.
        Raises FileNotFoundError if no index exists on disk and no documents given.
        """
        if documents is not None:
            self._build(documents)
        else:
            self._load()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _build(self, documents: List[Document]) -> None:
        self.documents = documents
        corpus = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        self._save()
        print(f"BM25 index built with {len(documents)} documents and saved to {INDEX_PATH}")

    def _save(self) -> None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)

    def _load(self) -> None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"No BM25 index found at {INDEX_PATH}. "
                "Run build_bm25_index() first."
            )
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        print(f"BM25 index loaded ({len(self.documents)} documents)")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Get top-k indices sorted by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.metadata["bm25_score"] = float(scores[idx])
            results.append(doc)
        return results

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[idx], float(scores[idx])) for idx in top_indices]


def build_bm25_index() -> BM25Retriever:
    """Fetch all documents from ChromaDB and build a fresh BM25 index."""
    from src.embeddings.vector_store import VectorStore
    from langchain.schema import Document

    store = VectorStore()
    total = store.count()
    print(f"Fetching {total} documents from ChromaDB...")

    # ChromaDB get() with no filter returns everything
    result = store.collection.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]

    return BM25Retriever(documents=documents)


if __name__ == "__main__":
    retriever = build_bm25_index()
    print("\nTest query: 'chest pain NSTEMI troponin'")
    hits = retriever.retrieve("chest pain NSTEMI troponin", k=3)
    for i, doc in enumerate(hits):
        print(f"\n[{i+1}] score={doc.metadata['bm25_score']:.3f} | {doc.metadata.get('disease','')}")
        print(f"     {doc.page_content[:200]}")

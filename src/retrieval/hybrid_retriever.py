"""
Hybrid retriever: BM25 + semantic search, fused then reranked.

Pipeline:
  1. BM25 retrieves top-k keyword matches
  2. ChromaDB retrieves top-k semantic matches
  3. Results are merged and deduplicated by chunk_id
  4. Cross-encoder reranker scores all candidates
  5. Top-k_rerank results are returned with citations
"""
from pathlib import Path
from typing import List

from langchain.schema import Document
from sentence_transformers import CrossEncoder

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    TOP_K_BM25,
    TOP_K_SEMANTIC,
    TOP_K_RERANK,
    RERANKER_MODEL,
)
from src.embeddings.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever, INDEX_PATH


class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()

        # Load BM25 index from disk (must be built first via build_bm25_index())
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                "BM25 index not found. Run:\n"
                "  python src/retrieval/bm25_retriever.py"
            )
        self.bm25 = BM25Retriever()

        print(f"Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve(self, query: str) -> List[Document]:
        # 1. BM25
        bm25_hits = self.bm25.retrieve(query, k=TOP_K_BM25)

        # 2. Semantic
        semantic_hits = self.vector_store.similarity_search(query, k=TOP_K_SEMANTIC)

        # 3. Merge + deduplicate by chunk_id
        seen: set = set()
        candidates: List[Document] = []
        for doc in bm25_hits + semantic_hits:
            cid = doc.metadata.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                candidates.append(doc)

        # 4. Rerank
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        # 5. Return top-k with rerank score in metadata
        results = []
        for score, doc in ranked[:TOP_K_RERANK]:
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)

        return results

    def retrieve_with_context(self, query: str) -> dict:
        """Return results plus a formatted context string ready for the LLM."""
        docs = self.retrieve(query)
        context_blocks = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("file_name", "unknown")
            disease = doc.metadata.get("disease", "")
            note_id = doc.metadata.get("note_id", "")
            citation = f"[{i+1}] {source}"
            if disease:
                citation += f" | {disease}"
            if note_id:
                citation += f" | {note_id}"
            context_blocks.append(f"{citation}\n{doc.page_content}")

        return {
            "query": query,
            "documents": docs,
            "context": "\n\n---\n\n".join(context_blocks),
        }


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "What are the diagnostic criteria for heart failure with preserved ejection fraction?"
    print(f"Query: {query}\n")
    result = retriever.retrieve_with_context(query)

    print(f"Retrieved {len(result['documents'])} documents:\n")
    for doc in result["documents"]:
        print(f"  score={doc.metadata['rerank_score']:.4f} | {doc.metadata.get('disease','')} | {doc.metadata.get('file_name','')}")
        print(f"  {doc.page_content[:200]}\n")

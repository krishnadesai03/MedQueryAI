"""
Main ingestion pipeline.

Loads all data sources, assigns globally unique chunk IDs,
and upserts everything into ChromaDB.

Usage:
    python src/ingestion/ingest.py [--sources pdf mimic]
"""
import argparse
import sys
from pathlib import Path
from typing import List

from langchain.schema import Document

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.document_loader import load_all_pdfs, load_all_mimic
from src.embeddings.vector_store import VectorStore


def assign_global_ids(chunks: List[Document]) -> List[Document]:
    """Replace per-source chunk_ids with globally unique sequential IDs."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def run(sources: List[str]) -> None:
    all_chunks: List[Document] = []

    if "pdf" in sources:
        print("=== Loading PDFs ===")
        pdf_chunks = load_all_pdfs()
        print(f"PDF chunks: {len(pdf_chunks)}\n")
        all_chunks.extend(pdf_chunks)

    if "mimic" in sources:
        print("=== Loading MIMIC-IV-Ext-DiReCT ===")
        mimic_chunks = load_all_mimic()
        print(f"MIMIC chunks: {len(mimic_chunks)}\n")
        all_chunks.extend(mimic_chunks)

    if not all_chunks:
        print("No chunks loaded. Check your data directories.")
        return

    all_chunks = assign_global_ids(all_chunks)
    print(f"Total chunks to ingest: {len(all_chunks)}\n")

    print("=== Storing in ChromaDB ===")
    store = VectorStore()
    store.embed_and_store(all_chunks)

    print(f"\nIngestion complete. Vectors in store: {store.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["pdf", "mimic"],
        default=["pdf", "mimic"],
        help="Data sources to ingest (default: all)",
    )
    args = parser.parse_args()
    run(args.sources)

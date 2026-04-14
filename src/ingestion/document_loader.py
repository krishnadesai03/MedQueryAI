import json
import os
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_DATA_DIR

MIMIC_DIR = Path(RAW_DATA_DIR) / "mimic-iv-ext-direct-1.0.0"
SAMPLES_DIR = MIMIC_DIR / "samples" / "Finished"
KG_DIR = MIMIC_DIR / "diagnostic_kg" / "Diagnosis_flowchart"

# Maps input keys to human-readable section names for readable chunks
_INPUT_LABELS = {
    "input1": "Chief Complaint",
    "input2": "History of Present Illness",
    "input3": "Past Medical History",
    "input4": "Medications",
    "input5": "Physical Examination",
    "input6": "Laboratory Results",
}


def clean_text(text: str) -> str:
    """Remove common PDF artifacts: excessive whitespace, page numbers, headers."""
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove lone page numbers (e.g. "- 3 -" or just "3" on its own line)
    text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF and return a list of cleaned Documents (one per page)."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    cleaned = []
    for page in pages:
        page.page_content = clean_text(page.page_content)
        # Enrich metadata
        page.metadata["file_name"] = Path(file_path).name
        page.metadata["file_path"] = str(file_path)
        if page.page_content:  # skip blank pages
            cleaned.append(page)

    return cleaned


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Tag each chunk with a unique ID for citation tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def _flatten_kg_knowledge(knowledge: dict, disease: str) -> str:
    """Recursively flatten a knowledge graph node into readable text."""
    lines = []
    for node, criteria in knowledge.items():
        if isinstance(criteria, dict):
            for aspect, detail in criteria.items():
                lines.append(f"{node} — {aspect}: {detail}")
        else:
            lines.append(f"{node}: {criteria}")
    return "\n".join(lines)


def load_mimic_knowledge_graphs() -> List[Document]:
    """Load DiReCT diagnostic knowledge graphs as Documents (one per disease)."""
    if not KG_DIR.exists():
        print(f"Knowledge graph directory not found: {KG_DIR}")
        return []

    documents = []
    for kg_file in sorted(KG_DIR.glob("*.json")):
        disease = kg_file.stem
        with open(kg_file, encoding="utf-8") as f:
            data = json.load(f)

        knowledge_text = _flatten_kg_knowledge(data.get("knowledge", {}), disease)
        if not knowledge_text.strip():
            continue

        content = f"Disease: {disease}\n\nDiagnostic Knowledge:\n{knowledge_text}"
        documents.append(Document(
            page_content=content,
            metadata={
                "source": "mimic-iv-ext-direct",
                "doc_type": "knowledge_graph",
                "disease": disease,
                "file_name": kg_file.name,
                "file_path": str(kg_file),
            }
        ))

    print(f"Loaded {len(documents)} knowledge graph documents")
    return documents


def _parse_diagnosis_key(key: str) -> tuple[str, str]:
    """Extract (diagnosis, step) from keys like 'HFpEF$Intermedia_5'."""
    parts = key.split("$", 1)
    diagnosis = parts[0].strip()
    step = parts[1].strip() if len(parts) > 1 else ""
    return diagnosis, step


def load_mimic_clinical_notes() -> List[Document]:
    """Load DiReCT annotated clinical notes as Documents."""
    if not SAMPLES_DIR.exists():
        print(f"Samples directory not found: {SAMPLES_DIR}")
        return []

    documents = []
    for json_file in sorted(SAMPLES_DIR.rglob("*.json")):
        # Infer disease category and optional sub-category from directory structure
        rel_parts = json_file.relative_to(SAMPLES_DIR).parts
        disease_category = rel_parts[0]
        sub_category = rel_parts[1] if len(rel_parts) == 3 else None

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Extract clinical text sections (input1–input6)
        sections = []
        for key, label in _INPUT_LABELS.items():
            value = data.get(key, "").strip()
            if value:
                sections.append(f"### {label}\n{value}")

        if not sections:
            continue

        # Extract final diagnosis from the top-level non-input key
        diagnosis_key = next(
            (k for k in data if not k.startswith("input")), None
        )
        final_diagnosis, _ = _parse_diagnosis_key(diagnosis_key) if diagnosis_key else ("Unknown", "")

        note_id = json_file.stem  # e.g. "11386787-DS-17"
        content = f"Disease: {disease_category}\nDiagnosis: {final_diagnosis}\n\n" + "\n\n".join(sections)

        metadata = {
            "source": "mimic-iv-ext-direct",
            "doc_type": "clinical_note",
            "disease": disease_category,
            "diagnosis": final_diagnosis,
            "note_id": note_id,
            "file_name": json_file.name,
            "file_path": str(json_file),
        }
        if sub_category:
            metadata["sub_category"] = sub_category

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Loaded {len(documents)} clinical note documents")
    return documents


def load_all_mimic(chunk: bool = True) -> List[Document]:
    """Load all MIMIC-IV-Ext-DiReCT data (notes + knowledge graphs) and optionally chunk."""
    docs = load_mimic_clinical_notes() + load_mimic_knowledge_graphs()
    if chunk:
        return chunk_documents(docs)
    return docs


def load_all_pdfs(directory: str = RAW_DATA_DIR) -> List[Document]:
    """Load and chunk all PDFs found in the given directory."""
    pdf_files = list(Path(directory).glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {directory}")
        return []

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        pages = load_pdf(str(pdf_path))
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)
        print(f"  -> {len(pages)} pages, {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # PDFs
    pdf_chunks = load_all_pdfs()
    if pdf_chunks:
        print("\nSample PDF chunk:")
        print(f"  Content : {pdf_chunks[0].page_content[:200]}")
        print(f"  Metadata: {pdf_chunks[0].metadata}")

    # MIMIC
    mimic_chunks = load_all_mimic()
    if mimic_chunks:
        print(f"\nTotal MIMIC chunks: {len(mimic_chunks)}")
        print("\nSample MIMIC chunk:")
        print(f"  Content : {mimic_chunks[0].page_content[:300]}")
        print(f"  Metadata: {mimic_chunks[0].metadata}")

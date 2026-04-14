"""
LLM Generation layer.

Takes retrieved + reranked context from HybridRetriever and calls Claude
to produce a grounded, cited answer.

Rules enforced via system prompt:
  - Answer using ONLY the provided context passages
  - Every claim must cite its source using [1], [2], ... inline
  - If the context does not contain enough information, respond with a
    safe refusal rather than guessing
  - Answers are for informational/clinical-decision-support purposes only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import anthropic
from langchain.schema import Document

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import LLM_MODEL, ANTHROPIC_API_KEY

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical decision-support assistant. Your sole purpose is to answer
healthcare questions using the evidence passages provided to you.

Rules you must follow without exception:
1. Use ONLY the context passages below to construct your answer.
2. Cite every claim inline using the passage number in square brackets, e.g. [1] or [2].
3. If the passages do not contain enough information to answer the question,
   respond exactly with:
   "I don't have enough information in the provided sources to answer this question."
4. Never speculate, invent facts, or use knowledge outside the provided passages.
5. End your answer with a "Sources" section listing each cited passage.
6. This system is for informational purposes only. Always include the disclaimer:
   "Note: This is for informational use only. Consult a qualified clinician for medical decisions."
"""

CONTEXT_TEMPLATE = "[{idx}] {header}\n{content}"

USER_TEMPLATE = """\
Context passages:

{context}

---

Question: {question}

Answer (cite sources inline with [1], [2], ...):
"""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[dict] = field(default_factory=list)
    refused: bool = False          # True when the LLM said "I don't know"

    def __str__(self) -> str:
        lines = [f"Q: {self.question}", "", f"A: {self.answer}"]
        if self.sources:
            lines += ["", "Sources:"]
            for s in self.sources:
                lines.append(
                    f"  [{s['idx']}] {s['disease']} | {s['file_name']} | {s['doc_type']}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class RAGGenerator:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = LLM_MODEL

    def _build_context(self, documents: List[Document]) -> tuple[str, list[dict]]:
        """Format retrieved docs into numbered context blocks + source metadata."""
        blocks = []
        sources = []
        for idx, doc in enumerate(documents, start=1):
            meta = doc.metadata
            disease  = meta.get("disease", "")
            doc_type = meta.get("doc_type", "")
            file_name = meta.get("file_name", "unknown")
            note_id  = meta.get("note_id", "")

            # Build a short readable header for the passage
            header_parts = [f"Source {idx}"]
            if disease:
                header_parts.append(disease)
            if note_id:
                header_parts.append(note_id)
            elif file_name:
                header_parts.append(file_name)
            if doc_type:
                header_parts.append(f"({doc_type})")
            header = " | ".join(header_parts)

            blocks.append(CONTEXT_TEMPLATE.format(
                idx=idx,
                header=header,
                content=doc.page_content.strip(),
            ))
            sources.append({
                "idx": idx,
                "disease": disease,
                "doc_type": doc_type,
                "file_name": file_name,
                "note_id": note_id,
                "rerank_score": meta.get("rerank_score", None),
            })

        return "\n\n---\n\n".join(blocks), sources

    def answer(self, question: str, documents: List[Document]) -> RAGResponse:
        """Generate a cited answer given a question and retrieved documents."""
        if not documents:
            return RAGResponse(
                question=question,
                answer="I don't have enough information in the provided sources to answer this question.",
                refused=True,
            )

        context_str, sources = self._build_context(documents)
        user_message = USER_TEMPLATE.format(
            context=context_str,
            question=question,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,   # deterministic — critical for medical use
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )

        answer_text = response.content[0].text.strip()
        refused = "i don't have enough information" in answer_text.lower()

        return RAGResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            refused=refused,
        )


# ---------------------------------------------------------------------------
# Convenience function — full pipeline in one call
# ---------------------------------------------------------------------------

def ask(question: str) -> RAGResponse:
    """
    End-to-end RAG: retrieve → rerank → generate.
    Initialises HybridRetriever and RAGGenerator on each call (fine for scripts;
    for a server you'd keep these as singletons).
    """
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    result = retriever.retrieve_with_context(question)
    documents = result["documents"]

    generator = RAGGenerator()
    return generator.answer(question, documents)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = [
        "What are the diagnostic criteria for heart failure with preserved ejection fraction?",
        "What symptoms are associated with COPD exacerbation?",
        "What is the capital of France?",   # out-of-scope — should be refused
    ]

    for q in questions:
        print("=" * 70)
        response = ask(q)
        print(response)
        print()

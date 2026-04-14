# MedQueryAI: Retrieval-Augmented Clinical Assistant

---

## What Is This Project?

Imagine a doctor who needs a quick, reliable answer to a complex medical question — like "What is the first-line treatment for heart failure with preserved ejection fraction?" Instead of spending 20 minutes searching through textbooks and guidelines, they type the question and get a cited, accurate answer in seconds.

That is what this project builds: a **Healthcare Question-Answering System** powered by AI. It reads medical documents, stores them intelligently, and when asked a question, finds the most relevant passages and uses an AI language model to write a clear answer — with references to the exact sources it used.

The key feature: **every claim is backed by a source**. The system is designed to never guess. If it doesn't know, it says so.

---

## The Big Picture — Full Pipeline

```
                        HEALTHCARE RAG PIPELINE
  ================================================================

  [RAW DATA]                  [INDEXING]               [ANSWERING]
  ----------                  ----------               -----------

  Clinical Notes  ──┐
  Knowledge Graphs ─┤──► Chunk & Clean ──► Embed ──► ChromaDB (Vector Store)
  Guidelines PDFs ──┘                        │              │
                                             │              │
                                             └──► BM25 Index│
                                                            │
                              User asks a question ─────────┤
                                                            │
                                              BM25 Search ──┤
                                          Semantic Search ──┼──► Merge & Deduplicate
                                                            │
                                                    Cross-Encoder Reranker
                                                            │
                                                    Top-K Results
                                                            │
                                            Claude (Anthropic API)
                                                            │
                                               Answer + [Source References]
                                                            │
                                                    Chat UI (Browser)
```

---

## How It Works — In Plain English

1. We start by loading raw medical data — 511 de-identified clinical notes and 24 disease knowledge graphs from the dataset
2. We split them into overlapping text chunks of ~800 tokens each
3. Each chunk is then converted into a 768-dimensional vector and stored in ChromaDB
4. Simultaneously a BM25 keyword index is built and saved to disk — together these form the knowledge base
5. A doctor types: *"What are the symptoms of a COPD exacerbation?"*
6. FastAPI receives it via a POST /ask request and passes it to the Hybrid Retriever, which runs two searches in parallel: BM25 for exact keyword matches (good for precise medical terms like "LVEF" or "troponin") and ChromaDB for semantic matches (good for meaning-based similarity like "heart attack" = "myocardial infarction")
7. Results from both searches are merged, deduplicated, and scored by a cross-encoder reranker that picks the top 3 most relevant chunks
8. Those chunks, with source metadata, are sent to Claude
9. Claude reads those top chunks and writes a clear, structured answer only from provided context
10. The response is then returned to the browser as a structured JSON with the answer, sources and a refusal flag
11. The answer looks like:

> *"During a COPD exacerbation, patients typically experience worsening dyspnea, increased cough, and sputum production [1]. Chest tightness and decreased exercise tolerance are also commonly reported [2]. In some cases, low oxygen saturation (e.g., 88% on room air) may be observed [3]."*
>
> *Sources:*
> *[1] clinical_note | COPD | 18591903-DS-16.json*
> *[2] knowledge_graph | COPD | COPD.json*
> *[3] clinical_note | COPD | 13227028-DS-14.json*
>
> *Note: This is for informational use only. Consult a qualified clinician for medical decisions.*

---

## Three Phases — The Roadmap

The project follows a 3-phase plan from the blueprint.

### Phase 1 — Prototype
**Goal:** Get a working demo that can answer basic medical questions with cited sources.

| Task |
|------|
| Set up Python environment and project structure | 
| Download and extract MIMIC-IV-Ext-DiReCT dataset |
| Build document loader for PDFs + MIMIC JSON notes + knowledge graphs |
| Chunk all documents (~800 tokens, 100 overlap) |
| Embed 3,833 chunks using `all-mpnet-base-v2` and store in ChromaDB |

**Deliverable:** 3,833 vectors indexed and searchable in ChromaDB.

---

### Phase 2 — Production Ready
**Goal:** Make the system smarter, faster, and accessible via a web interface.

| Task |
|------|
| Build BM25 keyword index (saved to disk) | Done |
| Build Hybrid Retriever (BM25 + semantic search + deduplication) | Done |
| Add cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) | Done |
| Build LLM generation layer with citation enforcement (Claude API) | Done |
| Safe refusal when question is out of scope or unsupported | Done |
| FastAPI server with `/ask` endpoint | Done |
| Chat UI served at `http://localhost:8000` | Done |

**Deliverable:** A fully working RAG web app — type a question, get a cited answer.

---

### Phase 3 — Evaluation & Testing 
**Goal:** Prove the system works with measurable scores and automated tests.

| Task |
|------|
| Build golden QA test set (20 questions across 8 diseases + 2 out-of-scope) | Done |
| Retrieval tests — recall@3 for every in-scope question | Done |
| Generation tests — keyword grounding, citation presence, disclaimer, refusal | Done |
| All 78 tests passing (56 generation + 22 retrieval) | Done |

**Test Results:**

| Test Suite | Tests | Result |
|---|---|---|
| Retrieval recall@3 | 22 | 100% |
| Answer contains keywords | 18 | 100% |
| Answer contains citations | 18 | 100% |
| Answer contains disclaimer | 18 | 100% |
| Out-of-scope refused | 2 | 100% |
| **Total** | **78** | **100%** |

**Deliverable:** Full automated test suite. Run anytime with `pytest tests/evaluation/ -v`.

---

## How to Run

### Start the server
```bash
venv/Scripts/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Open the chat UI
Go to **http://localhost:8000** in your browser.

### Run all tests
```bash
pytest tests/evaluation/ -v
```

### Re-ingest data (if you add new documents)
```bash
venv/Scripts/python src/ingestion/ingest.py
venv/Scripts/python src/retrieval/bm25_retriever.py
```

---

## Tech Stack

| Purpose | Tool / Library | Simple Description |
|---------|---------------|-------------------|
| Language | Python 3.11 | The programming language used |
| Document Loading | LangChain + PyPDFLoader | Reads PDFs and splits text into chunks |
| Embeddings | SentenceTransformers (`all-mpnet-base-v2`) | Converts text into numbers that capture meaning |
| Vector Store | ChromaDB | Database that stores and searches those number-vectors |
| Keyword Search | rank_bm25 | Classic search engine algorithm for exact keyword matching |
| Reranker | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Scores how relevant each result is to the question |
| LLM | Claude (`claude-sonnet-4-6`) via Anthropic API | Writes the final cited answer |
| API | FastAPI + Uvicorn | Web server that exposes the system over HTTP |
| Chat UI | Plain HTML/CSS/JavaScript | Browser-based chat interface, no framework needed |
| Testing | pytest | Runs all retrieval and generation quality tests |
| Dataset | MIMIC-IV-Ext-DiReCT (PhysioNet) | De-identified clinical notes used as knowledge base |

---

## Key Constraints

- **HIPAA Compliance:** All data is already de-identified (MIMIC-IV). No real patient identifiers are stored or sent externally.
- **Citation Required:** Every answer must cite its source. The system says "I don't know" rather than guessing.
- **Privacy by Design:** PHI never leaves the local environment.
- **Out-of-scope Refusal:** Questions about diseases not in the dataset (e.g., Malaria, COVID) or completely unrelated topics are refused rather than hallucinated.

---

## Expanding the Project in the Future

| Goal | What to do |
|------|-----------|
| Add more diseases | Download more data, run `ingest.py`, rebuild BM25 index, add new questions to `golden_qa.json` |
| Add clinical guidelines | Drop PDFs into `data/raw/`, re-run `ingest.py` |
| Improve answer quality | Switch to a biomedical embedding model (e.g., `Bio_ClinicalBERT`) |
| Switch LLM | Update `LLM_MODEL` in `.env` |
| Check for regressions | Run `pytest tests/evaluation/ -v` after any change |
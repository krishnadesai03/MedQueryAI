"""
Fetches PubMed abstracts using NCBI E-utilities API.
No API key required, but adding one (free) raises rate limit from 3 to 10 req/sec.
Get a free key at: https://www.ncbi.nlm.nih.gov/account/
"""

import time
import json
import os
from pathlib import Path
from typing import List, Dict
from urllib import request, parse, error

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import RAW_DATA_DIR

# Optional: set your NCBI API key in .env as NCBI_API_KEY
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Medical topics to fetch — covers the core healthcare RAG use cases
DEFAULT_QUERIES = [
    "type 2 diabetes treatment guidelines",
    "hypertension management clinical guidelines",
    "heart failure diagnosis treatment",
    "chronic kidney disease management",
    "pneumonia diagnosis treatment antibiotics",
    "sepsis management protocol",
    "antibiotic resistance clinical",
    "cancer screening guidelines",
    "depression treatment antidepressants",
    "asthma COPD management",
]


def search_pubmed(query: str, max_results: int = 50) -> List[str]:
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    url = f"{BASE_URL}/esearch.fcgi?{parse.urlencode(params)}"
    try:
        with request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data["esearchresult"]["idlist"]
    except error.URLError as e:
        print(f"  Search error for '{query}': {e}")
        return []


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """Fetch abstract text and metadata for a list of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "rettype": "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    url = f"{BASE_URL}/efetch.fcgi?{parse.urlencode(params)}"
    try:
        with request.urlopen(url, timeout=15) as resp:
            raw = resp.read().decode()
    except error.URLError as e:
        print(f"  Fetch error: {e}")
        return []

    # efetch with retmode=json for abstracts returns PubmedArticleSet
    # Use esummary for cleaner JSON metadata instead
    params["rettype"] = "docsum"
    url = f"{BASE_URL}/esummary.fcgi?{parse.urlencode(params)}"
    try:
        with request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except error.URLError as e:
        print(f"  Summary fetch error: {e}")
        return []

    articles = []
    result = data.get("result", {})
    for pmid in pmids:
        item = result.get(pmid, {})
        if not item or "error" in item:
            continue
        articles.append({
            "pmid": pmid,
            "title": item.get("title", ""),
            "authors": ", ".join(
                a.get("name", "") for a in item.get("authors", [])[:3]
            ),
            "journal": item.get("fulljournalname", item.get("source", "")),
            "pub_date": item.get("pubdate", ""),
            "source": f"PubMed PMID:{pmid}",
        })
    return articles


def fetch_abstract_text(pmids: List[str]) -> Dict[str, str]:
    """Fetch the actual abstract text for a list of PMIDs via efetch."""
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "text",
        "rettype": "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    url = f"{BASE_URL}/efetch.fcgi?{parse.urlencode(params)}"
    try:
        with request.urlopen(url, timeout=15) as resp:
            raw_text = resp.read().decode(errors="replace")
    except error.URLError as e:
        print(f"  Abstract text fetch error: {e}")
        return {}

    # Split into individual abstracts (separated by blank lines + numbering)
    blocks = raw_text.strip().split("\n\n\n")
    abstract_map = {}
    for i, pmid in enumerate(pmids):
        if i < len(blocks):
            abstract_map[pmid] = blocks[i].strip()
    return abstract_map


def save_articles(articles: List[Dict], abstract_texts: Dict[str, str], topic: str) -> str:
    """Save fetched articles as a text file in data/raw/pubmed/."""
    out_dir = Path(RAW_DATA_DIR) / "pubmed"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_topic = topic.replace(" ", "_").replace("/", "-")[:50]
    out_path = out_dir / f"{safe_topic}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# PubMed Articles: {topic}\n\n")
        for article in articles:
            pmid = article["pmid"]
            f.write(f"## {article['title']}\n")
            f.write(f"Authors: {article['authors']}\n")
            f.write(f"Journal: {article['journal']} ({article['pub_date']})\n")
            f.write(f"Source: {article['source']}\n\n")
            abstract = abstract_texts.get(pmid, "Abstract not available.")
            f.write(f"{abstract}\n\n")
            f.write("---\n\n")

    return str(out_path)


def fetch_topic(query: str, max_results: int = 50) -> str:
    """Full pipeline: search → fetch metadata + abstracts → save."""
    print(f"\nFetching: '{query}'")
    pmids = search_pubmed(query, max_results)
    if not pmids:
        print("  No results found.")
        return ""

    print(f"  Found {len(pmids)} articles. Fetching abstracts...")
    articles = fetch_abstracts(pmids)
    abstract_texts = fetch_abstract_text(pmids)

    out_path = save_articles(articles, abstract_texts, query)
    print(f"  Saved {len(articles)} articles -> {out_path}")

    time.sleep(0.4)  # Be respectful to NCBI rate limits (3 req/sec without key)
    return out_path


if __name__ == "__main__":
    print("Starting PubMed data fetch...")
    print(f"Topics to fetch: {len(DEFAULT_QUERIES)}")
    print(f"Articles per topic: 50")
    print(f"Output directory: {RAW_DATA_DIR}/pubmed/\n")

    saved_files = []
    for query in DEFAULT_QUERIES:
        path = fetch_topic(query, max_results=50)
        if path:
            saved_files.append(path)

    print(f"\nDone! Saved {len(saved_files)} topic files.")
    print("Next: run the ingestion pipeline to chunk and embed these files.")

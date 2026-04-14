"""
Downloads publicly available clinical guideline PDFs from NIH, CDC, and WHO.
All sources are free and require no login.
"""

import time
from pathlib import Path
from urllib import request, error
from typing import List, Dict

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import RAW_DATA_DIR

# Publicly available guideline PDFs — no login required
GUIDELINES = [
    # --- Diabetes ---
    {
        "name": "ADA_Standards_of_Care_2024",
        "url": "https://diabetesjournals.org/care/article-pdf/47/Supplement_1/S1/744762/dc24s001.pdf",
        "topic": "diabetes",
        "source": "American Diabetes Association 2024",
    },
    # --- Hypertension ---
    {
        "name": "JNC8_Hypertension_Guidelines",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4014893/pdf/jama_james_2013_oi_130010.pdf",
        "topic": "hypertension",
        "source": "JNC8 2014",
    },
    # --- HIV ---
    {
        "name": "NIH_HIV_Treatment_Guidelines",
        "url": "https://clinicalinfo.hiv.gov/sites/default/files/guidelines/documents/adult-adolescent-arv/guidelines-adult-adolescent-arv.pdf",
        "topic": "HIV",
        "source": "NIH AIDSinfo 2024",
    },
    # --- Asthma ---
    {
        "name": "NHLBI_Asthma_Guidelines",
        "url": "https://www.nhlbi.nih.gov/sites/default/files/media/docs/asthma-guidelines-2020.pdf",
        "topic": "asthma",
        "source": "NHLBI 2020",
    },
    # --- Sepsis ---
    {
        "name": "Surviving_Sepsis_Campaign_2021",
        "url": "https://link.springer.com/content/pdf/10.1007/s00134-021-06506-y.pdf",
        "topic": "sepsis",
        "source": "Surviving Sepsis Campaign 2021",
    },
    # --- CDC Opioid Prescribing ---
    {
        "name": "CDC_Opioid_Prescribing_Guidelines_2022",
        "url": "https://www.cdc.gov/mmwr/volumes/71/rr/pdfs/rr7103a1-H.pdf",
        "topic": "opioids",
        "source": "CDC 2022",
    },
    # --- Cholesterol / Cardiovascular ---
    {
        "name": "ACC_AHA_Cholesterol_Guidelines_2018",
        "url": "https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000000625",
        "topic": "cardiovascular",
        "source": "ACC/AHA 2018",
    },
    # --- COVID-19 ---
    {
        "name": "NIH_COVID19_Treatment_Guidelines",
        "url": "https://files.covid19treatmentguidelines.nih.gov/guidelines/covid19treatmentguidelines.pdf",
        "topic": "COVID-19",
        "source": "NIH 2024",
    },
]


def download_pdf(guideline: Dict, out_dir: Path) -> bool:
    """Download a single PDF guideline to disk."""
    topic_dir = out_dir / "guidelines" / guideline["topic"]
    topic_dir.mkdir(parents=True, exist_ok=True)

    out_path = topic_dir / f"{guideline['name']}.pdf"

    if out_path.exists():
        print(f"  Already exists, skipping: {out_path.name}")
        return True

    print(f"  Downloading: {guideline['name']}...")
    headers = {"User-Agent": "Mozilla/5.0 (research bot; healthcare RAG project)"}
    req = request.Request(guideline["url"], headers=headers)

    try:
        with request.urlopen(req, timeout=30) as resp:
            content = resp.read()

        # Basic check: PDFs start with %PDF
        if not content.startswith(b"%PDF"):
            print(f"  Warning: response doesn't look like a PDF for {guideline['name']}")
            return False

        with open(out_path, "wb") as f:
            f.write(content)

        size_kb = len(content) / 1024
        print(f"  Saved ({size_kb:.0f} KB): {out_path}")
        return True

    except error.HTTPError as e:
        print(f"  HTTP {e.code} error for {guideline['name']}: {e.reason}")
        return False
    except error.URLError as e:
        print(f"  URL error for {guideline['name']}: {e.reason}")
        return False
    except Exception as e:
        print(f"  Unexpected error for {guideline['name']}: {e}")
        return False


def download_all_guidelines(out_dir: str = RAW_DATA_DIR) -> List[str]:
    """Download all guidelines and return list of successfully saved paths."""
    out_path = Path(out_dir)
    print(f"Downloading {len(GUIDELINES)} clinical guidelines to: {out_path}/guidelines/\n")

    success, failed = [], []
    for guideline in GUIDELINES:
        ok = download_pdf(guideline, out_path)
        if ok:
            success.append(guideline["name"])
        else:
            failed.append(guideline["name"])
        time.sleep(1)  # Be polite to servers

    print(f"\nResults: {len(success)} downloaded, {len(failed)} failed.")
    if failed:
        print(f"Failed: {failed}")
        print("Note: Some URLs may have moved. Check the source website for updated links.")

    return success


if __name__ == "__main__":
    download_all_guidelines()
    print("\nNext: run the ingestion pipeline to chunk and embed the downloaded PDFs.")

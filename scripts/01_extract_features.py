"""
Script 01: Full feature extraction from candidates.jsonl.gz
Output: precomputed/features.parquet

Feature logic lives in pipeline/feature_extraction.py — shared with 04_eval.py.
Any change to extraction must be made there, not here.
"""

import gzip
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.feature_extraction import extract_features

DATA_PATH = ROOT / "data" / "candidates.jsonl.gz"
OUT_PATH  = ROOT / "precomputed" / "features.parquet"


def main():
    os.makedirs(OUT_PATH.parent, exist_ok=True)
    rows = []
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting features"):
            line = line.strip()
            if not line:
                continue
            rows.append(extract_features(json.loads(line), detect_honeypot=True))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUT_PATH}")
    print(f"Honeypots detected:                           {df['is_honeypot'].sum()}")
    print(f"IT-services-only careers:                     {df['entire_career_it_services'].sum()}")
    print(f"Research-only careers (no production):        {df['entire_career_research_only'].sum()}")
    print(f"Shallow recent-ML-only (LangChain pattern):   {df['shallow_recent_ml_only'].sum()}")


if __name__ == "__main__":
    main()

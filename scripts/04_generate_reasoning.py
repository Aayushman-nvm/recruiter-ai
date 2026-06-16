"""
Script 04: Generate reasoning for the exact top-100 candidates via Ollama phi3:mini.

Input:  precomputed/top100_ids.txt  (written by rank.py --no-reasoning)
        data/candidates.jsonl.gz
        jd.txt  (FULL job description — context richness helps here)
Output: precomputed/reasoning_cache.json

Run AFTER a first-pass of rank.py with --no-reasoning:
  python rank.py --no-reasoning
  python scripts/04_generate_reasoning.py
  python rank.py --out submission.csv    (final pass, picks up cache)

Ollama must be running: `ollama serve` in a separate terminal.
Model: phi3:mini (2.2GB) — pull with `ollama pull phi3:mini`

Bug fix: last_active_days_ago is a DERIVED value, not a raw redrob_signals field.
  The raw field is last_active_date (a date string).
  Always compute inline: (REFERENCE_DATE - date.fromisoformat(sig["last_active_date"])).days
  Never use sig.get("last_active_days_ago") — that key does not exist.
"""

import gzip
import json
import os
from datetime import date

import requests
from tqdm import tqdm

from scoring import IT_SERVICES_COMPANIES, REFERENCE_DATE

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"

DATA_PATH    = "data/candidates.jsonl.gz"
IDS_PATH     = "precomputed/top100_ids.txt"
OUT_PATH     = "precomputed/reasoning_cache.json"
JD_TEXT_PATH = "jd.txt"   # Full JD — NOT jd_query.txt

PROMPT_TEMPLATE = """\
You are a senior technical recruiter evaluating a candidate for a specific role.

JOB DESCRIPTION:
{jd_text}

CANDIDATE FACTS (do not invent anything not listed here):
- Name: {name}
- Current role: {title} at {company} ({company_type})
- Years of experience: {yoe}
- Location: {location}, {country} | Willing to relocate: {relocate}
- Career highlight (recent): {recent_role}
- Key skills: {skills}
- Notice period: {notice} days | Open to work: {open_to_work}
- Last active on platform: {days_ago} days ago
- Recruiter response rate: {response_rate}

Write exactly 2 sentences:
Sentence 1: What specifically makes this candidate a fit or not fit (cite their actual \
title, years of experience, and company type — no generic praise).
Sentence 2: One concrete concern or standout positive (cite a real number or fact above).

Rules: No generic phrases like "strong background" or "excellent fit".
       Never invent facts not listed above.
       Sentence structure must vary — each candidate's reasoning should read differently.\
"""


def classify_company_type(career: list) -> str:
    """Classify career as IT services or product/startup for context in prompt."""
    if not career:
        return "unknown"
    all_lower = [r.get("company", "").lower() for r in career]
    all_it = all(
        any(it in co for it in IT_SERVICES_COMPANIES)
        for co in all_lower
    ) if all_lower else False
    return "IT services only" if all_it else "product/startup company"


def main():
    os.makedirs("precomputed", exist_ok=True)

    with open(JD_TEXT_PATH, encoding="utf-8") as f:
        jd_text = f.read().strip()

    with open(IDS_PATH) as f:
        top100_ids = set(f.read().strip().split("\n"))

    print(f"Generating reasoning for {len(top100_ids)} candidates...")

    # Scan JSONL for only the top-100 profiles (early exit once all found)
    profiles = {}
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in top100_ids:
                profiles[c["candidate_id"]] = c
            if len(profiles) == len(top100_ids):
                break

    print(f"Found {len(profiles)}/{len(top100_ids)} profiles in JSONL.")

    reasoning_cache = {}

    for cid, c in tqdm(profiles.items(), desc="Generating reasoning"):
        p = c["profile"]
        sig = c.get("redrob_signals", {})
        career = c.get("career_history", [])

        # CRITICAL: compute last_active_days_ago inline — NOT from sig.get(...)
        try:
            last_active = date.fromisoformat(str(sig.get("last_active_date", ""))[:10])
            days_ago = (REFERENCE_DATE - last_active).days
        except Exception:
            days_ago = 999

        # Most recent role summary
        recent_role = ""
        if career:
            r = career[0]
            dur = r.get("duration_months", 0) or 0
            recent_role = f"{r.get('title', '')} at {r.get('company', '')} ({dur} months)"

        company_type = classify_company_type(career)
        skills_text = ", ".join(s.get("name", "") for s in c.get("skills", [])[:6])

        prompt = PROMPT_TEMPLATE.format(
            jd_text=jd_text,
            name=p.get("anonymized_name", "Candidate"),
            title=p.get("current_title", ""),
            company=p.get("current_company", ""),
            company_type=company_type,
            yoe=p.get("years_of_experience", 0),
            location=p.get("location", ""),
            country=p.get("country", ""),
            relocate=sig.get("willing_to_relocate", False),
            recent_role=recent_role,
            skills=skills_text,
            notice=sig.get("notice_period_days", 90),
            open_to_work=sig.get("open_to_work_flag", False),
            days_ago=days_ago,
            response_rate=round(float(sig.get("recruiter_response_rate", 0) or 0), 2),
        )

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.4, "num_predict": 100},
                },
                timeout=30,
            )
            text = response.json().get("response", "").strip()
        except Exception as e:
            print(f"  Warning: Ollama call failed for {cid}: {e}")
            text = ""

        reasoning_cache[cid] = text

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(reasoning_cache, f, indent=2)

    print(f"\nGenerated reasoning for {len(reasoning_cache)} candidates → {OUT_PATH}")


if __name__ == "__main__":
    main()

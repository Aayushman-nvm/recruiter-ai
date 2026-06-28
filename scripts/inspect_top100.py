"""
scripts/inspect_top100.py — Dump raw candidate data for top-100 submission entries.

Usage:
  python scripts/inspect_top100.py --submission test_submissions/submission.csv
  python scripts/inspect_top100.py --submission test_submissions/submission.csv --out-dir bin/inspection_sub

Output:
  bin/inspection/<submission_name>/
    00_summary.csv          — all 100 candidates with key fields side-by-side
    candidates/
      001_CAND_XXXXXXX.json — full raw JSON for each candidate (rank-prefixed)

⚠️  SUBMISSION WARNING: This script is for development inspection only.
    It writes to bin/ and is not part of the submission pipeline.
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "candidates.jsonl.gz"


def load_submission(csv_path: str) -> list[dict]:
    """Load submission CSV into list of {candidate_id, rank, score, reasoning}."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "candidate_id": row["candidate_id"],
                "rank":         int(row["rank"]),
                "score":        float(row["score"]),
                "reasoning":    row["reasoning"],
            })
    return sorted(rows, key=lambda r: r["rank"])


def build_index(target_ids: set[str]) -> dict[str, dict]:
    """
    Stream candidates.jsonl.gz once and collect only the target IDs.
    Much faster than reading the whole file into memory.
    """
    found = {}
    print(f"Scanning {DATA_PATH} for {len(target_ids)} candidates...")
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("candidate_id", "")
            if cid in target_ids:
                found[cid] = c
                if len(found) == len(target_ids):
                    break  # found everything, no need to keep reading
    return found


def extract_summary_row(rank_entry: dict, raw: dict) -> dict:
    """
    Pull the fields most relevant for manual inspection into a flat dict
    for the summary CSV.
    """
    p   = raw.get("profile", {})
    sig = raw.get("redrob_signals", {})
    career = raw.get("career_history", [])
    edu    = raw.get("education", [])

    # Last 3 roles for quick career scan
    recent_roles = "; ".join(
        f"{r.get('title','')} @ {r.get('company','')} ({r.get('duration_months',0) or 0}mo)"
        for r in career[:3]
    )

    # Skill names (top 8 by proficiency)
    PROF_ORDER = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills_sorted = sorted(
        raw.get("skills", []),
        key=lambda s: PROF_ORDER.get(s.get("proficiency", ""), 0),
        reverse=True,
    )
    top_skills = ", ".join(s.get("name", "") for s in skills_sorted[:8])

    edu_tier  = edu[0].get("tier", "")        if edu else ""
    edu_field = edu[0].get("field_of_study", "") if edu else ""

    return {
        "rank":                  rank_entry["rank"],
        "score":                 rank_entry["score"],
        "candidate_id":          rank_entry["candidate_id"],
        "current_title":         p.get("current_title", ""),
        "current_company":       p.get("current_company", ""),
        "years_of_experience":   p.get("years_of_experience", ""),
        "location":              p.get("location", ""),
        "country":               p.get("country", ""),
        "headline":              p.get("headline", ""),
        "notice_period_days":    sig.get("notice_period_days", ""),
        "open_to_work":          sig.get("open_to_work_flag", ""),
        "last_active_date":      sig.get("last_active_date", ""),
        "recruiter_response_rate": sig.get("recruiter_response_rate", ""),
        "willing_to_relocate":   sig.get("willing_to_relocate", ""),
        "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d", ""),
        "edu_tier":              edu_tier,
        "edu_field":             edu_field,
        "recent_roles":          recent_roles,
        "top_skills":            top_skills,
        "reasoning":             rank_entry["reasoning"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        default="test_submissions/submission_11.csv",
        help="Path to the submission CSV to inspect.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to bin/inspection_<submission_stem>/",
    )
    args = parser.parse_args()

    submission_path = Path(args.submission)
    if not submission_path.exists():
        print(f"ERROR: {submission_path} not found.")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "bin" / f"inspection_{submission_path.stem}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load submission ───────────────────────────────────────────────────────
    entries = load_submission(str(submission_path))
    print(f"Loaded {len(entries)} entries from {submission_path.name}.")
    target_ids = {e["candidate_id"] for e in entries}

    # ── Load raw data ─────────────────────────────────────────────────────────
    raw_by_id = build_index(target_ids)
    missing = target_ids - set(raw_by_id.keys())
    if missing:
        print(f"WARNING: {len(missing)} candidate IDs not found in dataset: {missing}")

    # ── Write candidate JSON ──────────────────────────────────────────────
    print(f"Writing combined JSON file → {out_dir}")

    all_candidates = []

    for entry in entries:
        cid  = entry["candidate_id"]
        rank = entry["rank"]
        raw  = raw_by_id.get(cid)

        if raw is None:
            continue

        candidate = dict(raw)  # avoid mutating raw_by_id

        candidate["_submission"] = {
            "rank": rank,
            "score": entry["score"],
            "reasoning": entry["reasoning"],
        }

        all_candidates.append({
            "rank": entry["rank"],
            "candidate_id": cid,
            "score": entry["score"],
            "reasoning": entry["reasoning"],
            "candidate": raw,
        })

    combined_json_path = out_dir / "all_candidates.json"

    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)

    print(f"Combined JSON → {combined_json_path}")
    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_rows = []
    for entry in entries:
        cid = entry["candidate_id"]
        raw = raw_by_id.get(cid)
        if raw is None:
            summary_rows.append({"rank": entry["rank"], "candidate_id": cid,
                                  "score": entry["score"], "error": "NOT_FOUND"})
        else:
            summary_rows.append(extract_summary_row(entry, raw))

    summary_path = out_dir / "00_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV → {summary_path}")

    print(f"\nDone. Open {out_dir} to inspect:")
    print("  00_summary.csv              — all 100 candidates, key fields")
    print("  all_candidates.json  — full raw profiles of 100 candidates.")
    print("\n⚠️  SUBMISSION WARNING: bin/ is development only — not for submission.")


if __name__ == "__main__":
    main()

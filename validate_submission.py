"""
validate_submission.py — Sanity checks for submission.csv.

Usage: python validate_submission.py [submission.csv]

Checks:
  1. Exactly 100 rows
  2. Required columns present (candidate_id, rank, score, reasoning)
  3. Ranks 1–100, each used exactly once
  4. candidate_ids unique
  5. Scores non-increasing (rank 1 has highest score)
  6. No missing candidate_ids or scores
"""

import sys
import pandas as pd


def validate(path: str):
    print(f"Validating: {path}")
    errors = []
    warnings = []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"FATAL: Could not read CSV: {e}")
        sys.exit(1)

    # 1. Row count
    if len(df) != 100:
        errors.append(f"Row count: expected 100, got {len(df)}")
    else:
        print(f"  ✓ 100 rows")

    # 2. Required columns
    required = {"candidate_id", "rank", "score", "reasoning"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    else:
        print(f"  ✓ Required columns present")

    if errors:
        # Can't continue without required columns
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    # 3. Ranks 1–100, each exactly once
    ranks = sorted(df["rank"].tolist())
    if ranks != list(range(1, 101)):
        errors.append(f"Ranks are not exactly 1–100. Got: {ranks[:5]}...")
    else:
        print(f"  ✓ Ranks 1–100, each used exactly once")

    # 4. Unique candidate_ids
    if df["candidate_id"].nunique() != len(df):
        dupes = df[df["candidate_id"].duplicated()]["candidate_id"].tolist()
        errors.append(f"Duplicate candidate_ids: {dupes}")
    else:
        print(f"  ✓ candidate_ids unique")

    # 5. No null candidate_ids or scores
    if df["candidate_id"].isnull().any():
        errors.append("Null values in candidate_id column")
    if df["score"].isnull().any():
        errors.append("Null values in score column")
    if not errors:
        print(f"  ✓ No null candidate_ids or scores")

    # 6. Scores non-increasing
    # Sort by rank to ensure we check in rank order
    df_sorted = df.sort_values("rank")
    scores = df_sorted["score"].tolist()
    violations = [
        (i + 1, scores[i], scores[i + 1])
        for i in range(len(scores) - 1)
        if scores[i] < scores[i + 1]
    ]
    if violations:
        errors.append(
            f"Score not non-increasing at {len(violations)} positions. "
            f"First violation: rank {violations[0][0]} "
            f"(score {violations[0][1]:.6f}) < rank {violations[0][0]+1} "
            f"(score {violations[0][2]:.6f})"
        )
    else:
        print(f"  ✓ Scores monotonically non-increasing")

    # 7. Reasoning column populated (warning, not error)
    n_empty_reasoning = df["reasoning"].fillna("").eq("").sum()
    if n_empty_reasoning > 0:
        warnings.append(f"{n_empty_reasoning}/100 candidates have empty reasoning strings")
    else:
        print(f"  ✓ All 100 candidates have reasoning strings")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if warnings:
        for w in warnings:
            print(f"  ⚠ Warning: {w}")
    if errors:
        print(f"\n❌ VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print("✅ All checks passed. Submission is valid.")
        print(f"\nTop 5 candidates:")
        top5 = df.sort_values("rank").head(5)[["rank", "candidate_id", "score"]]
        print(top5.to_string(index=False))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    validate(path)

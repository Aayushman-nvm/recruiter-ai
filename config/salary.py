"""
config/salary.py — Estimated salary target range for this JD.

Role: Senior AI Engineer, Series A India startup, 5–9 YoE, AI specialisation.
Wide range deliberately chosen to reduce false negatives from estimation error.
The JD itself does not state a compensation band — this is an estimate used
only as a soft tiebreaker (weight: 0.02 in compute_structural_score).
"""

SALARY_TARGET_MIN = 20.0   # INR LPA
SALARY_TARGET_MAX = 65.0   # INR LPA

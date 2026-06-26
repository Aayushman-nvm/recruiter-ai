"""
scoring/reasoning.py — generate_reasoning() function.

Produces two-sentence recruiter-facing reasoning for each candidate.
Sentence 1: lead signal (disqualifiers first, then strongest positive).
Sentence 2: concrete secondary fact that qualifies or reinforces sentence 1.

Design principles:
  - Concise: recruiters review 100 candidates; every word must earn its place.
  - Analytical: sentence 2 surfaces specific measurable flags, not boilerplate.
  - Location + relocation always present for non-disqualified candidates.
  - Priority order: hard disqualifiers > narrative evidence > logistics > tiebreakers.
"""

from config.salary import SALARY_TARGET_MIN, SALARY_TARGET_MAX


def generate_reasoning(row: dict) -> str:
    # ── Extract row fields ────────────────────────────────────────────────────
    title         = str(row.get("current_title", "candidate") or "candidate").strip()
    yoe           = float(row.get("years_of_experience", 0) or 0)
    company       = str(row.get("current_company", "") or "").strip()
    location      = str(row.get("location", "") or "").strip()
    country       = str(row.get("country", "") or "").strip()

    is_india         = bool(row.get("is_india_based", False))
    is_target_city   = bool(row.get("is_target_city", False))
    willing_relocate = bool(row.get("willing_to_relocate", False))
    entire_it        = bool(row.get("entire_career_it_services", False))
    entire_research  = bool(row.get("entire_career_research_only", False))
    shallow_ml_only  = bool(row.get("shallow_recent_ml_only", False))
    has_product      = bool(row.get("has_product_company_exp", False))
    has_ml           = bool(row.get("has_ml_production_experience", False))
    yrs_since_ml     = float(row.get("years_since_last_ml_role") if row.get("years_since_last_ml_role") is not None else 99)
    traj_score       = float(row.get("trajectory_score") if row.get("trajectory_score") is not None else 0)
    avg_tenure       = float(row.get("avg_tenure_months") if row.get("avg_tenure_months") is not None else 0)
    trajectory_up    = bool(row.get("trajectory_upward", False))

    sal_min       = float(row.get("salary_min_lpa") if row.get("salary_min_lpa") is not None else 0)
    sal_max       = float(row.get("salary_max_lpa") if row.get("salary_max_lpa") is not None else 999)
    open_to_work  = bool(row.get("open_to_work_flag", False))
    days_ago      = int(row.get("last_active_days_ago") if row.get("last_active_days_ago") is not None else 999)
    response_rate = float(row.get("recruiter_response_rate") if row.get("recruiter_response_rate") is not None else 0)
    notice_days   = int(row.get("notice_period_days") if row.get("notice_period_days") is not None else 90)
    resp_time_hrs = float(row.get("avg_response_time_hours") if row.get("avg_response_time_hours") is not None else -1)
    saves         = int(row.get("saved_by_recruiters_30d") if row.get("saved_by_recruiters_30d") is not None else 0)
    skill_bonus   = float(row.get("skill_assessment_bonus") if row.get("skill_assessment_bonus") is not None else 0)
    edu_tier      = str(row.get("edu_tier", "unknown") or "unknown")
    ce_score      = float(row.get("ce_score") if row.get("ce_score") is not None else 0)
    n_it          = int(row.get("n_it_services_roles", 0) or 0)

    has_disqualifying_lang = bool(row.get("has_disqualifying_language", False))
    n_ghost                = int(row.get("n_ghost_skills", 0) or 0)
    is_ghost               = bool(row.get("is_ghost_skill_candidate", False))
    is_cv_speech_nlp       = bool(row.get("is_cv_speech_no_nlp", False))

    # ── Derived strings ───────────────────────────────────────────────────────
    yoe_str      = f"{yoe:.1f}"
    loc_str      = f"{location}, {country}" if location and country else (location or country or "unknown location")
    company_str  = company if company else "their current employer"
    relocate_str = "open to relocation" if willing_relocate else "not open to relocation"
    notice_str   = f"{notice_days}d notice"

    # ── Pre-compute condition flags ───────────────────────────────────────────
    location_negative = is_india and not is_target_city and not willing_relocate
    abroad_candidate  = not is_india
    notice_long       = notice_days > 90

    # ── Sentence 1: Lead signal ───────────────────────────────────────────────

    if has_disqualifying_lang and not entire_it and not entire_research:
        loc_ctx = location if location else loc_str
        s1 = (f"{title}, {yoe_str}y exp, {company_str} ({loc_ctx}, {relocate_str}, {notice_str}). "
              f"Career narrative explicitly states production deployment was handled by another team — not an end-to-end owner.")

    elif entire_it:
        s1 = (f"{title}, {yoe_str}y exp — entire career in IT services consulting, "
              f"which the JD explicitly disqualifies ({company_str}, {notice_str}).")

    elif entire_research:
        s1 = (f"{title}, {yoe_str}y exp — career entirely in research/academic roles, "
              f"no production deployment detected. JD is explicit: not a fit.")

    elif shallow_ml_only:
        s1 = (f"{title}, {yoe_str}y exp at {company_str}. Only recent (<12mo) LangChain/API-layer "
              f"ML detected — JD flags this without prior production ML depth as insufficient.")

    elif not has_ml and not has_product:
        s1 = (f"{title}, {yoe_str}y exp at {company_str} ({notice_str}). "
              f"No production ML experience and no product-company background detected.")

    elif has_ml and yrs_since_ml <= 1:
        if yoe > 15:
            s1 = (f"{title}, {yoe_str}y exp, currently at {company_str} in production ML. "
                  f"At {yoe_str} years total, JD flags elevated risk of having shifted from hands-on coding to architecture.")
        else:
            loc_ctx = location if location else loc_str
            s1 = (f"{title}, {yoe_str}y exp, currently at {company_str} in production ML "
                  f"({loc_ctx}, {relocate_str}, {notice_str}).")

    elif has_ml and 1 < yrs_since_ml <= 3:
        s1 = (f"{title}, {yoe_str}y exp — last production ML role {yrs_since_ml:.1f}y ago at {company_str}. "
              f"Relevant background, but recency gap is a concern for a hands-on coding role.")

    elif has_ml and yrs_since_ml > 3:
        s1 = (f"{title}, {yoe_str}y exp — most recent ML role was {yrs_since_ml:.1f}y ago. "
              f"JD requires active hands-on ML, not historical experience.")

    elif has_product and not has_ml:
        s1 = (f"{title}, {yoe_str}y exp at product companies incl. {company_str}. "
              f"No production ML/retrieval/ranking detected in career history.")

    elif ce_score >= 0.7:
        s1 = (f"{title}, {yoe_str}y exp at {company_str} — strong semantic fit with JD "
              f"(cross-encoder: {ce_score:.2f}). {location}, {relocate_str}.")

    else:
        s1 = (f"{title}, {yoe_str}y exp at {company_str} ({notice_str}, {relocate_str}).")

    # ── Sentence 2: Supporting / qualifying signal ────────────────────────────

    if entire_it or entire_research or shallow_ml_only:
        activity = f"last active {days_ago}d ago" if days_ago < 999 else "activity unknown"
        s2 = (f"Response rate {response_rate:.0%}, {activity}. "
              f"{'Open to work. ' if open_to_work else 'Not actively looking. '}"
              f"Notice: {notice_days}d.")

    elif has_disqualifying_lang:
        notes = []
        if n_ghost >= 3:
            notes.append(f"{n_ghost} JD skills claimed but absent from narrative (ghost skills)")
        if notice_days > 60:
            notes.append(f"{notice_days}d notice above JD's 30d buyout window")
        if not is_india:
            notes.append(f"abroad ({loc_str}{', willing to relocate' if willing_relocate else ', not open to relocation'})")
        elif not is_target_city and not willing_relocate:
            notes.append(f"non-preferred city ({location}), not open to relocation")
        if days_ago > 60:
            notes.append(f"last active {days_ago}d ago")
        s2 = ("Additional flags: " + "; ".join(notes) + ".") if notes else (
            f"Response rate {response_rate:.0%}, last active {days_ago}d ago. Notice: {notice_days}d."
        )

    elif is_cv_speech_nlp:
        s2 = ("Primary domain is CV/speech with insufficient NLP/IR crossover in narrative. "
              "JD explicitly flags this: 'you'd be re-learning fundamentals here.'")

    elif is_ghost:
        s2 = (f"{n_ghost} JD-relevant skills listed (e.g. vector DBs, semantic search) "
              f"but absent from career narrative — keyword stuffing likely. Ghost-skill penalty applied.")

    elif abroad_candidate:
        if willing_relocate:
            s2 = (f"Based in {loc_str}, willing to self-fund relocation. "
                  f"No visa sponsorship — handled case-by-case per JD. Notice: {notice_days}d.")
        else:
            s2 = (f"Based in {loc_str}, not willing to relocate. "
                  f"Significant location mismatch for Noida/Pune role. Notice: {notice_days}d.")

    elif notice_long and location_negative:
        it_note = f" {n_it} IT-services role(s) in history." if n_it >= 1 else ""
        s2 = (f"{notice_days}d notice + non-preferred city ({location}, not open to relocation) "
              f"— two compounding friction points.{it_note}")

    elif notice_long:
        it_note = f" {n_it} IT-services role(s) in history." if n_it >= 1 else ""
        s2 = (f"{notice_days}d notice exceeds JD's 30d buyout window.{it_note} "
              f"Last active {days_ago}d ago, response rate {response_rate:.0%}.")

    elif n_it >= 1 and not entire_it:
        severity = "minor" if n_it <= 2 else ("moderate" if n_it <= 4 else "strong")
        s2 = (f"{n_it} IT-services role(s) in career history — {severity} penalty. "
              f"Location: {location}, {relocate_str}. Notice: {notice_days}d.")

    elif location_negative:
        s2 = (f"Non-preferred city ({location}), not open to relocation. "
              f"Notice: {notice_days}d, last active {days_ago}d ago.")

    elif skill_bonus >= 0.04:
        s2 = (f"Platform-verified assessments on JD-relevant skills (bonus: {skill_bonus:.3f}). "
              f"Location: {location}, {relocate_str}. Notice: {notice_days}d.")

    elif traj_score >= 0.8 and trajectory_up:
        tenure_str = f"{avg_tenure:.0f}mo avg tenure" if avg_tenure > 0 else ""
        s2 = (f"Upward title trajectory, {tenure_str} — stable, not title-chasing. "
              f"Location: {location}, {relocate_str}. Notice: {notice_days}d.")

    elif avg_tenure < 15 and avg_tenure > 0:
        s2 = (f"Avg tenure {avg_tenure:.0f}mo/role — possible title-chasing (JD concern). "
              f"Location: {location}, {relocate_str}. Notice: {notice_days}d.")

    elif open_to_work and days_ago <= 14 and response_rate >= 0.7:
        s2 = (f"Actively job-seeking: open to work, last active {days_ago}d ago, "
              f"{response_rate:.0%} response rate — highly reachable.")

    elif open_to_work and notice_days <= 30:
        start_str = "immediately" if notice_days == 0 else f"within {notice_days}d"
        s2 = (f"Open to work, can start {start_str}. "
              f"Last active {days_ago}d ago, {response_rate:.0%} response rate.")

    elif days_ago > 180:
        s2 = (f"Last active {days_ago}d ago ({response_rate:.0%} response rate) — "
              f"low platform engagement raises reachability concerns.")

    elif resp_time_hrs >= 0 and resp_time_hrs <= 4:
        s2 = (f"Responds within {resp_time_hrs:.0f}h on average "
              f"({response_rate:.0%} response rate) — strong engagement signal.")

    elif saves >= 5:
        s2 = (f"Saved by {saves} recruiters in last 30 days — crowd-validated interest. "
              f"Location: {location}, {relocate_str}.")

    elif sal_min > SALARY_TARGET_MAX * 1.3:
        s2 = (f"Expected salary (min {sal_min:.0f} LPA) likely above Series A budget — "
              f"offer negotiation risk.")

    elif sal_max < SALARY_TARGET_MIN * 0.7:
        s2 = (f"Expected range ({sal_min:.0f}–{sal_max:.0f} LPA) below market rate for this seniority.")

    elif edu_tier == "tier_1":
        s2 = (f"Tier-1 institution background — marginal tiebreaker. "
              f"Location: {location}, {relocate_str}. Notice: {notice_days}d.")

    else:
        active_str = "recently" if days_ago <= 30 else f"{days_ago}d ago"
        s2 = (f"Last active {active_str}, {response_rate:.0%} response rate, {notice_days}d notice. "
              f"Location: {location}, {relocate_str}.")

    return f"{s1} {s2}"

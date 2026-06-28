"""
scoring/reasoning.py — generate_reasoning() function.

Produces two tight sentences for each candidate.
Sentence 1: strongest signal (what makes or breaks this candidate).
Sentence 2: the most important logistical or qualifying fact.

Rank-aware: top-10 leads with the positive; mid/tail leads with the dominant flag.
No fixed template — sentence structure varies by candidate profile.
"""



def generate_reasoning(row: dict) -> str:
    # ── Extract fields ────────────────────────────────────────────────────────
    title         = str(row.get("current_title", "candidate") or "candidate").strip()
    yoe           = float(row.get("years_of_experience", 0) or 0)
    company       = str(row.get("current_company", "") or "").strip()
    location      = str(row.get("location", "") or "").strip()
    country       = str(row.get("country", "") or "").strip()
    rank          = int(row.get("rank", 99) or 99)

    is_india         = bool(row.get("is_india_based", False))
    is_primary_city  = bool(row.get("is_primary_city", False))
    is_tier_1_city   = bool(row.get("is_tier_1_city", False))
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

    open_to_work  = bool(row.get("open_to_work_flag", False))
    days_ago      = int(row.get("last_active_days_ago") if row.get("last_active_days_ago") is not None else 999)
    response_rate = float(row.get("recruiter_response_rate") if row.get("recruiter_response_rate") is not None else 0)
    notice_days   = int(row.get("notice_period_days") if row.get("notice_period_days") is not None else 90)
    sal_min       = float(row.get("salary_min_lpa") if row.get("salary_min_lpa") is not None else 0)
    sal_max       = float(row.get("salary_max_lpa") if row.get("salary_max_lpa") is not None else 999)
    n_it          = int(row.get("n_it_services_roles", 0) or 0)

    has_disqualifying_lang = bool(row.get("has_disqualifying_language", False))
    n_ghost                = int(row.get("n_ghost_skills", 0) or 0)
    is_ghost               = bool(row.get("is_ghost_skill_candidate", False))
    is_cv_speech_nlp       = bool(row.get("is_cv_speech_no_nlp", False))
    is_junior_stagnant     = bool(row.get("is_junior_stagnant", False))
    top_jd_skills          = str(row.get("top_jd_skills", "") or "").strip()
    ce_score               = float(row.get("ce_score") if row.get("ce_score") is not None else 0)

    # ── Derived helpers ───────────────────────────────────────────────────────
    company_str = company if company else "their current company"
    loc_str     = location if location else country

    # Location context string — concise
    if is_primary_city:
        loc_ctx = f"{location} (primary city)"
    elif is_tier_1_city and willing_relocate:
        loc_ctx = f"{location}, willing to relocate"
    elif is_tier_1_city:
        loc_ctx = f"{location}, NOT willing to relocate"
    elif willing_relocate:
        loc_ctx = f"{location}, willing to relocate"
    elif not is_india:
        loc_ctx = f"{location}, {country}" if location and country else (location or country)
    else:
        loc_ctx = f"{location}, not willing to relocate"

    notice_str = f"{notice_days}d notice" if notice_days > 30 else (
        "immediately available" if notice_days == 0 else f"{notice_days}d notice"
    )

    skills_str = f" Skills: {top_jd_skills}." if top_jd_skills else ""

    # ── Hard disqualifier cases — these dominate regardless of rank ───────────
    if entire_it:
        s1 = f"{title} ({yoe:.1f}y) — entire career in IT services consulting, which the JD explicitly disqualifies."
        s2 = f"Current company: {company_str}. {notice_str}."
        return f"{s1} {s2}"

    if entire_research:
        s1 = f"{title} ({yoe:.1f}y) — career entirely in research/academic roles with no production deployment."
        s2 = f"JD is explicit: production experience required. {notice_str}."
        return f"{s1} {s2}"

    if shallow_ml_only:
        s1 = f"{title} ({yoe:.1f}y) at {company_str} — only recent (<12mo) API-layer ML detected, no pre-LLM-era production depth."
        s2 = f"JD specifically flags this pattern as insufficient. {loc_ctx}, {notice_str}."
        return f"{s1} {s2}"

    if is_cv_speech_nlp:
        s1 = f"{title} ({yoe:.1f}y) — primary domain is CV/speech with insufficient NLP/IR crossover in narrative."
        s2 = f"JD explicitly excludes this: 're-learning fundamentals.' {loc_ctx}, {notice_str}."
        return f"{s1} {s2}"

    # ── Trajectory summary helper (used in positive cases) ───────────────────
    if avg_tenure > 24 and trajectory_up:
        traj_str = f"stable upward trajectory ({avg_tenure:.0f}mo avg tenure)"
    elif avg_tenure < 15 and avg_tenure > 0:
        traj_str = f"short avg tenure ({avg_tenure:.0f}mo/role — possible title-chasing)"
    elif trajectory_up:
        traj_str = "upward trajectory"
    else:
        traj_str = None

    # ── Reachability flag — JD explicitly names low response rate as disqualifying ──
    low_response = response_rate < 0.20
    low_response_str = f"{response_rate:.0%} response rate" if low_response else None

    # ── TOP 10: lead with what's good, notice/logistics secondary ────────────
    if rank <= 10:
        if has_ml and yrs_since_ml <= 1:
            ml_ctx = "currently in production ML"
        elif has_ml and yrs_since_ml <= 2:
            ml_ctx = f"production ML {yrs_since_ml:.1f}y ago"
        elif has_ml:
            ml_ctx = f"ML background ({yrs_since_ml:.1f}y since last ML role)"
        else:
            ml_ctx = "product-company background"

        disq_note = " Narrative flags non-ownership of deployment." if has_disqualifying_lang else ""
        ghost_note = f" {n_ghost} ghost skills flagged." if is_ghost else ""
        junior_note = f" Title stuck at junior level for {sum([1]) * int(yoe * 12):.0f}mo — no promotion detected." if is_junior_stagnant else ""
        reach_note = f" Low response rate ({response_rate:.0%}) — reachability concern." if low_response else ""

        s1 = f"{title}, {yoe:.1f}y exp at {company_str} — {ml_ctx}.{skills_str}{disq_note}{ghost_note}{junior_note}{reach_note}"

        logistics = []
        if is_primary_city:
            logistics.append(f"based in {location} (office city)")
        else:
            logistics.append(loc_ctx)
        logistics.append(notice_str)
        if traj_str:
            logistics.append(traj_str)
        s2 = ", ".join(logistics).capitalize() + "."
        return f"{s1} {s2}"

    # ── MID RANGE (11–50): lead with dominant signal, flag the main friction ──
    if rank <= 50:
        if has_disqualifying_lang:
            s1 = f"{title} ({yoe:.1f}y, {company_str}) — narrative states deployment was handled by another team.{skills_str}"
            flags = []
            if notice_days > 60:
                flags.append(notice_str)
            if not is_india:
                flags.append(f"abroad ({loc_ctx})")
            elif not is_primary_city and not willing_relocate:
                flags.append(f"{loc_ctx}")
            if is_junior_stagnant:
                flags.append("junior title with no promotion")
            if low_response:
                flags.append(low_response_str)
            s2 = ("Flags: " + ", ".join(flags) + ".") if flags else f"{loc_ctx}, {notice_str}."
        elif is_junior_stagnant:
            s1 = f"{title}, {yoe:.1f}y — junior-level title throughout career with no detected promotion.{skills_str}"
            s2 = f"Mismatch for a founding-team role requiring senior judgment. {loc_ctx}, {notice_str}."
        elif has_ml and yrs_since_ml <= 1:
            s1 = f"{title}, {yoe:.1f}y — active production ML at {company_str}.{skills_str}"
            frictions = []
            if notice_days > 60:
                frictions.append(notice_str)
            if not is_primary_city and not willing_relocate:
                frictions.append(f"{loc_ctx}")
            elif not is_primary_city:
                frictions.append(loc_ctx)
            if n_it >= 1:
                frictions.append(f"{n_it} IT-services role(s)")
            if low_response:
                frictions.append(low_response_str)
            s2 = (", ".join(frictions) + ".").capitalize() if frictions else f"{loc_ctx}, {notice_str}."
        elif has_ml and yrs_since_ml <= 3:
            s1 = f"{title}, {yoe:.1f}y — last ML role {yrs_since_ml:.1f}y ago at {company_str}, recency gap is a concern.{skills_str}"
            s2 = f"{loc_ctx}, {notice_str}."
        else:
            s1 = f"{title}, {yoe:.1f}y at {company_str} — {'product-company background' if has_product else 'no ML/product background detected'}.{skills_str}"
            s2 = f"{loc_ctx}, {notice_str}."
        return f"{s1} {s2}"

    # ── TAIL (51–100): honest about friction, brief on positives ─────────────
    if has_disqualifying_lang:
        s1 = f"{title} ({yoe:.1f}y) — narrative explicitly states production deployment owned by a separate team.{skills_str}"
        flags = []
        if notice_days > 90:
            flags.append(notice_str)
        if not is_india:
            flags.append(f"abroad ({loc_ctx})")
        elif not is_primary_city and not willing_relocate:
            flags.append(f"{loc_ctx}")
        if n_ghost >= 3:
            flags.append(f"{n_ghost} ghost skills")
        if is_junior_stagnant:
            flags.append("junior title, no promotion")
        if low_response:
            flags.append(low_response_str)
        s2 = ("Additional flags: " + ", ".join(flags) + ".") if flags else f"{loc_ctx}, {notice_str}."
    elif is_junior_stagnant:
        s1 = f"{title}, {yoe:.1f}y — junior title throughout career, no upward movement detected.{skills_str}"
        s2 = f"Significant seniority mismatch for founding-team role. {loc_ctx}, {notice_str}."
    elif has_ml:
        s1 = f"{title}, {yoe:.1f}y at {company_str} — ML background, {yrs_since_ml:.1f}y since last active ML role.{skills_str}"
        extras = []
        if low_response:
            extras.append(low_response_str)
        s2 = f"{loc_ctx}, {notice_str}" + (", " + ", ".join(extras) if extras else "") + "."
    else:
        s1 = f"{title}, {yoe:.1f}y at {company_str} — no production ML detected in career narrative.{skills_str}"
        extras = []
        if low_response:
            extras.append(low_response_str)
        s2 = f"{loc_ctx}, {notice_str}" + (", " + ", ".join(extras) if extras else "") + "."

    return f"{s1} {s2}"

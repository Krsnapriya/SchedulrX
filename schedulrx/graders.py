"""
SchedulrX Dual Grading System
================================
Two complementary graders:

1. Programmatic Grader (deterministic, always available)
   - Constraint satisfaction
   - Soft constraint reasoning
   - Adaptability (replanning after cancellation)
   - Step efficiency

2. LLM Grader (optional, requires ANTHROPIC_API_KEY)
   - Fairness assessment
   - Naturalness of scheduling
   - Reasoning quality

Combined grader blends both when available.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import pytz


# ─────────────────────────────────────────────────────────────────────────────
# Programmatic Grader (deterministic — Phase 3 weapon)
# ─────────────────────────────────────────────────────────────────────────────

def programmatic_grade(
    requests: list,
    scheduled: list,
    profiles: dict,
    profiles_read: dict,
    participant_schedules: dict,
    step_count: int,
    max_steps: int,
    cancelled_meetings: list = None,
    participants: dict = None,
    episode_start_time: datetime = None,
) -> Dict:
    """
    Structured grader with capability scores and failure mode detection.

    Returns:
        {
            "score": float,
            "capabilities": {
                "constraint_satisfaction": float,
                "soft_constraint_reasoning": float,
                "adaptability": float,
                "efficiency": float,
            },
            "failure_modes": [str, ...],
            "trajectory_summary": {
                "steps": int,
                "replans": int,
                "conflicts_detected": bool,
                "profiles_explored": int,
            }
        }
    """
    cancelled_meetings = cancelled_meetings or []
    participants = participants or {}
    failure_modes = []
    n_requests = len(requests)

    if n_requests == 0:
        return {"score": 0.0, "capabilities": {}, "failure_modes": [], "trajectory_summary": {}}

    # ── 1. Constraint Satisfaction (Hard) ────────────────────────────────
    # Active meetings = total - cancelled
    active_ids = set()
    for r in requests:
        rid = r.id if hasattr(r, "id") else r.get("id", "")
        if rid not in cancelled_meetings:
            active_ids.add(rid)

    scheduled_ids = {m["meeting_id"] for m in scheduled}
    n_active = len(active_ids)
    n_scheduled = len(scheduled_ids & active_ids)

    completion_score = n_scheduled / max(n_active, 1)
    if completion_score < 1.0:
        unscheduled = active_ids - scheduled_ids
        failure_modes.append(f"Failed to schedule: {sorted(unscheduled)}")

    # Check dependency violations
    mid_times = {m["meeting_id"]: m["time"] for m in scheduled}
    dep_violations = 0
    for r in requests:
        rid = r.id if hasattr(r, "id") else r.get("id", "")
        dep = r.depends_on if hasattr(r, "depends_on") else r.get("depends_on")
        if dep and rid in mid_times and dep in mid_times:
            t1 = datetime.fromisoformat(mid_times[rid].replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(mid_times[dep].replace("Z", "+00:00"))
            if t1 <= t2:
                dep_violations += 1
                failure_modes.append(f"Dependency violation: {rid} scheduled before {dep}")

    constraint_score = max(0.0, completion_score - dep_violations * 0.3)

    # Check deadline violations
    for r in requests:
        rid = r.id if hasattr(r, "id") else r.get("id", "")
        deadline = r.deadline_hours if hasattr(r, "deadline_hours") else r.get("deadline_hours")
        if deadline and rid in mid_times and episode_start_time:
            cutoff = episode_start_time + __import__("datetime").timedelta(hours=deadline)
            t = datetime.fromisoformat(mid_times[rid].replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = pytz.utc.localize(t)
            if t > cutoff:
                constraint_score -= 0.2
                failure_modes.append(f"Deadline missed: {rid}")

    constraint_score = max(0.0, min(1.0, constraint_score))

    # ── 2. Soft Constraint Reasoning ─────────────────────────────────────
    # Did the agent read profiles before scheduling? Did it respect preferences?
    soft_score = 1.0
    soft_violations = 0

    for m in scheduled:
        for pid in m.get("participants", []):
            # Critical: did agent read profile before scheduling?
            if pid not in profiles_read:
                soft_violations += 1
                if f"Scheduled without reading {pid}'s profile" not in failure_modes:
                    failure_modes.append(f"Scheduled without reading {pid}'s profile")

            # Did it respect preferred times?
            prof = profiles.get(pid)
            if prof and pid in profiles_read and m.get("time"):
                prefs = prof.preferred_times if hasattr(prof, "preferred_times") else prof.get("preferred_times", [])
                try:
                    mt = datetime.fromisoformat(m["time"].replace("Z", "+00:00"))
                    p_data = participants.get(pid)
                    if p_data:
                        tz_str = p_data.timezone if hasattr(p_data, "timezone") else p_data.get("timezone", "UTC")
                        local_h = mt.astimezone(pytz.timezone(tz_str)).hour
                        pref_match = False
                        for p in prefs:
                            if p == "morning" and 6 <= local_h < 12:
                                pref_match = True
                            elif p == "afternoon" and 12 <= local_h < 17:
                                pref_match = True
                            elif p == "evening" and 17 <= local_h < 21:
                                pref_match = True
                        if not pref_match and prefs:
                            soft_violations += 1
                            failure_modes.append(f"Ignored {pid}'s preference ({prefs}) for {m['meeting_id']}")
                except (ValueError, AttributeError):
                    pass

    total_checks = max(sum(len(m.get("participants", [])) for m in scheduled), 1)
    soft_score = max(0.0, 1.0 - (soft_violations * 0.15))

    # ── 3. Adaptability ─────────────────────────────────────────────────
    # Did the agent replan after cancellations?
    adapt_score = 1.0
    replans = 0

    if cancelled_meetings:
        rescheduled = set()
        for m in scheduled:
            if m["meeting_id"] in cancelled_meetings:
                rescheduled.add(m["meeting_id"])
                replans += 1

        unrescheduled = set(cancelled_meetings) - rescheduled
        if unrescheduled:
            adapt_score = 0.3
            failure_modes.append(f"Did not replan after cancellation: {sorted(unrescheduled)}")
        else:
            adapt_score = 1.0  # Successfully replanned
    else:
        adapt_score = 1.0  # No cancellations to handle

    # ── 4. Step Efficiency ───────────────────────────────────────────────
    # Only reward efficiency if task is completed
    if n_scheduled == n_active and n_active > 0:
        efficiency_score = max(0.0, 1.0 - (step_count / max(max_steps, 1)))
    else:
        efficiency_score = 0.0

    # ── Composite Score ──────────────────────────────────────────────────
    weights = {
        "constraint_satisfaction": 0.35,
        "soft_constraint_reasoning": 0.25,
        "adaptability": 0.20,
        "efficiency": 0.20,
    }

    capabilities = {
        "constraint_satisfaction": round(constraint_score, 3),
        "soft_constraint_reasoning": round(soft_score, 3),
        "adaptability": round(adapt_score, 3),
        "efficiency": round(efficiency_score, 3),
    }

    final_score = sum(weights[k] * capabilities[k] for k in weights)

    # ── Trajectory Summary ───────────────────────────────────────────────
    trajectory_summary = {
        "steps": step_count,
        "replans": replans,
        "conflicts_detected": len(cancelled_meetings) > 0,
        "profiles_explored": len(profiles_read),
        "total_participants": len(participants),
        "meetings_scheduled": n_scheduled,
        "meetings_active": n_active,
    }

    return {
        "score": round(final_score, 4),
        "capabilities": capabilities,
        "failure_modes": failure_modes,
        "trajectory_summary": trajectory_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Grader (optional — requires ANTHROPIC_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

def llm_grade(
    scenario: str,
    proposed_slot: str,
    outcome: str,
) -> Dict:
    """
    LLM-based quality scoring via Claude. Evaluates naturalness,
    fairness, and contextual appropriateness.

    Gracefully falls back to a neutral score if no API key is present.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return {
            "fairness": 0.5,
            "naturalness": 0.5,
            "efficiency": 0.5,
            "composite": 0.5,
            "reasoning": "LLM grader unavailable (no ANTHROPIC_API_KEY). Using neutral fallback.",
            "source": "fallback",
        }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are an expert meeting-scheduling evaluator.

Scenario:
{scenario}

Proposed slot: {proposed_slot}
Outcome: {outcome}

Score the scheduling decision on a scale of 0.0–1.0 across three dimensions:
1. fairness       — how equitably the slot distributes inconvenience
2. naturalness    — how much the slot aligns with human working norms
3. efficiency     — how quickly a good slot was found

Reply ONLY in this JSON format (no markdown):
{{"fairness": 0.0, "naturalness": 0.0, "efficiency": 0.0, "reasoning": "..."}}"""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        result = json.loads(msg.content[0].text)
        result["composite"] = round(
            0.4 * result["fairness"]
            + 0.35 * result["naturalness"]
            + 0.25 * result["efficiency"],
            4,
        )
        result["source"] = "claude"
        return result

    except Exception as e:
        return {
            "fairness": 0.5,
            "naturalness": 0.5,
            "efficiency": 0.5,
            "composite": 0.5,
            "reasoning": f"LLM grader error: {str(e)[:100]}",
            "source": "fallback",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Combined Grader
# ─────────────────────────────────────────────────────────────────────────────

def combined_grade(
    programmatic_result: Dict,
    scenario: str = "",
    proposed_slot: str = "",
    outcome: str = "",
) -> Dict:
    """
    Combine programmatic (deterministic) and LLM (optional) graders.
    Programmatic is always authoritative. LLM adds bonus insight.
    """
    prog_score = programmatic_result.get("score", 0.0)

    llm_result = llm_grade(scenario, proposed_slot, outcome)
    llm_score = llm_result.get("composite", 0.5)

    # Weighting: 70% programmatic (deterministic), 30% LLM (bonus)
    # If LLM is fallback, its weight is effectively 0 (score=0.5 is neutral)
    if llm_result.get("source") == "fallback":
        final = prog_score  # Pure programmatic
    else:
        final = 0.7 * prog_score + 0.3 * llm_score

    return {
        "final_score": round(final, 4),
        "programmatic": programmatic_result,
        "llm": llm_result,
    }

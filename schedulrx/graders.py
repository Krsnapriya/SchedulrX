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

def get_final_slot(trajectory: List[Dict]) -> Optional[str]:
    """Extract the very last proposed slot for a meeting to enforce decision locking."""
    for step in reversed(trajectory):
        action = step.get("action", {})
        # Depending on format: action can be a dict or a string representation
        if isinstance(action, dict):
            if action.get("type") == "propose_slot" or action.get("type") == "schedule_meeting":
                return action.get("slot") or action.get("proposed_time")
        elif isinstance(action, str):
            # Fallback for string-represented actions (e.g., from logs)
            if "schedule_meeting" in action or "propose_slot" in action:
                # Basic extraction as fallback
                import re
                match = re.search(r'slot=[\'"]([^\'"]+)[\'"]', action)
                if match: return match.group(1)
    return None

def detect_recovery(trajectory: List[Dict]) -> bool:
    """Detect if an agent noticed a mistake and corrected it (Recovery)."""
    violated = False
    for step in trajectory:
        info = step.get("info", {})
        if info.get("soft_constraint_violation"):
            violated = True
        if violated and info.get("soft_constraint_satisfied"):
            return True
    return False

def detect_soft_constraint_usage(trajectory: List[Dict], metrics: Dict) -> float:
    """
    Trajectory-aware soft constraint reasoning score.
    Logic:
    1.0: Perfect Success (Directly avoided trap)
    0.7: Recovery (Detected and fixed violation)
    0.5: Partial (Mentioned/Noticed but failed to correct final decision)
    0.0: Ignored / Blind Failure
    """
    violation = metrics.get("soft_constraint_violations", 0) > 0 if metrics else False
    success = metrics.get("soft_constraint_successes", 0) > 0 if metrics else False
    recovery = detect_recovery(trajectory)

    mentioned = any(
        any(k in str(step.get("action", "")).lower() for k in ["preference", "avoid", "profile", "limit"])
        for step in trajectory
    )

    if success and not violation:
        return 1.0
    elif recovery:
        return 0.7
    elif mentioned:
        return 0.5
    return 0.2 if success else 0.0

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
    metrics: dict = None,
    trajectory: List[Dict] = None,
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
        return {"score": 0.001, "capabilities": {}, "failure_modes": [], "trajectory_summary": {}}

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

    # ── 2. Soft Constraint Reasoning (Audit Hardening) ──────────────────
    trajectory = trajectory or []
    metrics = metrics or {}
    
    # Decision Locking: Grade the agent on its FINAL choice
    final_slot = get_final_slot(trajectory)
    # Smoke test: 09:00 is the known trap in adv_001
    trap_slot = "09:00" 
    
    hard_fail = False
    if final_slot and trap_slot in final_slot:
        hard_fail = True
    
    if final_slot is None and n_scheduled > 0:
        # Fallback if trajectory format changed
        final_slot = scheduled[-1].get("time") if scheduled else None

    # Strict Hierarchy Scoring
    soft_score = detect_soft_constraint_usage(trajectory, metrics)
    
    # Dominance: Final failure (trap) wipes the reasoning score
    if hard_fail:
        soft_score = 0.0
        failure_modes.append("Critical Failure: Final decision triggered a known trap (09:00)")

    # ── 3. Adaptability (Dynamic & Cancellation) ─────────────────────────
    replans = 0
    adaptation_score = 1.0
    
    # Check for Dynamic Preference Shift (Level 3)
    dynamic_update_step = -1
    for i, step in enumerate(trajectory):
        if "dynamic_update" in step.get("info", {}):
            dynamic_update_step = step.get("step", -1)
            break
    
    if dynamic_update_step != -1:
        # User Rule: 1-Step Grace Period
        grace_step = dynamic_update_step + 1
        adapted = False
        
        # Did agent reschedule AFTER grace step?
        for step in trajectory:
            curr_step = step.get("step", -1)
            if curr_step > grace_step and step.get("action", {}).get("action_type") == "reschedule_meeting":
                adapted = True
                break
        
        if not adapted:
            adaptation_score = 0.3
            failure_modes.append("Failed to adapt to mid-episode preference shift (Level 3)")
        else:
            insight = "Agent successfully adapted and replanned after a dynamic preference shift."

    # Check for Stochastic Cancellations
    if cancelled_meetings:
        rescheduled = set()
        for m in scheduled:
            if m["meeting_id"] in cancelled_meetings:
                rescheduled.add(m["meeting_id"])
        
        unrescheduled = set(cancelled_meetings) - rescheduled
        if unrescheduled:
            adaptation_score = min(adaptation_score, 0.5)
            failure_modes.append(f"Did not replan after cancellation: {sorted(unrescheduled)}")

    # ── 4. Trade-off Reasoning (Level 2) ────────────────────────────────
    tradeoff_score = 1.0
    # Search for L2 conflict triggers in trajectory
    for step in trajectory:
        info = step.get("info", {})
        if info.get("soft_violation_details") and len(info["soft_violation_details"]) > 1:
            # Agent chose a slot with > 1 penalty (Penalty 2)
            # In Level 2, 14:00 has Penalty 1, 09:00 has Penalty 2.
            tradeoff_score = 0.5
            failure_modes.append("Suboptimal trade-off: Agent chose a higher-penalty slot (Level 2)")
            break

    # ── 5. Step Efficiency ───────────────────────────────────────────────
    if n_scheduled == n_active and n_active > 0:
        efficiency_score = max(0.0, 1.0 - (step_count / max(max_steps, 1)))
    else:
        efficiency_score = 0.0

    # ── Composite Score & Capabilities ──────────────────────────────────
    capabilities = {
        "constraint_satisfaction": round(constraint_score, 3),
        "soft_constraint_reasoning": round(soft_score, 3),
        "tradeoff_reasoning": round(tradeoff_score, 3),
        "adaptability": round(adaptation_score, 3),
        "efficiency": round(efficiency_score, 3),
    }

    # Final Score Weighting (Adjusted for new categories)
    final_score = (
        capabilities["constraint_satisfaction"] * 0.3 +
        capabilities["soft_constraint_reasoning"] * 0.2 +
        capabilities["tradeoff_reasoning"] * 0.2 +
        capabilities["adaptability"] * 0.2 +
        capabilities["efficiency"] * 0.1
    )

    # Insight Extraction (Judge Utility)
    if soft_score == 1.0:
        insight = "Agent demonstrated perfect alignment with implicit human preferences."
    elif soft_score >= 0.7:
        insight = "Agent successfully identified and corrected a soft-constraint violation."
    elif hard_fail:
        insight = "Agent optimized for surface availability but failed to account for implicit human constraints."
    else:
        insight = "Agent completed task but failed to demonstrate reasoning depth."

    # ── Trajectory Summary ───────────────────────────────────────────────
    trajectory_summary = {
        "steps": step_count,
        "replans": replans,
        "conflicts_detected": len(cancelled_meetings) > 0,
        "profiles_explored": len(profiles_read),
        "total_participants": len(participants),
        "meetings_scheduled": n_scheduled,
    }

    return {
        "score": round(final_score, 4),
        "capabilities": capabilities,
        "adversarial_analysis": {
            "trap_slot": trap_slot,
            "final_slot": final_slot,
            "violation": hard_fail,
            "recovery_detected": detect_recovery(trajectory),
            "insight": insight
        },
        "failure_modes": failure_modes,
        "insight": insight,
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

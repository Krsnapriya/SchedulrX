"""
SchedulrX Multi-Objective Reward Decomposition
=================================================
Exposes all reward components individually so judges (and agents)
can see exactly what is being optimized.

Components:
  1. conflict_free          — availability overlap (hard constraint)
  2. preference_alignment   — match participant preferred_times
  3. timezone_fairness      — penalise antisocial local hours
  4. efficiency             — reward finding good slots earlier
  5. soft_constraint_penalty — hidden traps from profiles
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pytz


@dataclass
class RewardComponents:
    """Transparent reward breakdown for judges and diagnostics."""

    conflict_free: float = 0.0
    preference_alignment: float = 0.0
    timezone_fairness: float = 0.0
    efficiency: float = 0.0
    soft_constraint_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Weights — judges can audit these
REWARD_WEIGHTS = {
    "conflict_free": 0.35,
    "preference_alignment": 0.25,
    "timezone_fairness": 0.15,
    "efficiency": 0.10,
    "soft_constraint_penalty": 0.15,
}


def compute_reward(
    proposed_time: str,
    participants: List[str],
    participant_data: dict,
    profiles_read: dict,
    profiles: dict,
    scheduled: list,
    participant_schedules: dict,
    step_count: int,
    max_steps: int,
    duration_minutes: int = 60,
) -> Tuple[float, RewardComponents]:
    """
    Compute multi-objective reward with full component visibility.

    Returns:
        (total_reward, RewardComponents) — total is weighted sum, components
        are individual scores in [0, 1] (or negative for penalties).
    """
    components = RewardComponents()

    try:
        dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        components.total = -0.5
        return components.total, components

    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    end_dt = dt + __import__("datetime").timedelta(minutes=duration_minutes)

    # ── 1. Conflict-Free Score ──────────────────────────────────────────
    # How many participants are actually free at this slot?
    n_free = 0
    n_total = len(participants)

    for pid in participants:
        p = participant_data.get(pid)
        if not p:
            continue
        avail = p.get("availability") if isinstance(p, dict) else getattr(p, "availability", None)
        if avail is None:
            continue  # Availability unknown — neutral
        tz_str = p.get("timezone", "UTC") if isinstance(p, dict) else getattr(p, "timezone", "UTC")
        tz = pytz.timezone(tz_str)
        local_dt = dt.astimezone(tz)
        local_end = end_dt.astimezone(tz)

        for slot in avail:
            s_start = datetime.fromisoformat(slot["start"])
            s_end = datetime.fromisoformat(slot["end"])
            if s_start.tzinfo is None:
                s_start = pytz.utc.localize(s_start)
            if s_end.tzinfo is None:
                s_end = pytz.utc.localize(s_end)
            if local_dt >= s_start and local_end <= s_end:
                n_free += 1
                break

    components.conflict_free = n_free / max(n_total, 1)

    # ── 2. Preference Alignment ─────────────────────────────────────────
    # Do scheduled times match participant preferred_times?
    pref_scores = []
    hour = dt.hour

    for pid in participants:
        prof = profiles_read.get(pid) or profiles.get(pid)
        if not prof:
            pref_scores.append(0.5)  # Unknown — neutral
            continue

        prefs = prof.get("preferred_times") or [] if isinstance(prof, dict) else (getattr(prof, "preferred_times") or [])
        if not prefs:
            pref_scores.append(0.5)
            continue

        # Convert participant local hour
        p = participant_data.get(pid)
        tz_str = p.get("timezone", "UTC") if isinstance(p, dict) else getattr(p, "timezone", "UTC")
        local_hour = dt.astimezone(pytz.timezone(tz_str)).hour

        match = False
        for p_time in prefs:
            if p_time == "morning" and 6 <= local_hour < 12:
                match = True
            elif p_time == "afternoon" and 12 <= local_hour < 17:
                match = True
            elif p_time == "evening" and 17 <= local_hour < 21:
                match = True
        pref_scores.append(1.0 if match else 0.3)

    components.preference_alignment = sum(pref_scores) / max(len(pref_scores), 1)

    # ── 3. Timezone Fairness ────────────────────────────────────────────
    # Penalise slots that are antisocial for some participants (before 6am or after 9pm local)
    in_hours_count = 0
    for pid in participants:
        p = participant_data.get(pid)
        if not p:
            continue
        tz_str = p.get("timezone", "UTC") if isinstance(p, dict) else getattr(p, "timezone", "UTC")
        local_hour = dt.astimezone(pytz.timezone(tz_str)).hour
        if 6 <= local_hour <= 21:
            in_hours_count += 1

    components.timezone_fairness = in_hours_count / max(n_total, 1)

    # ── 4. Efficiency ───────────────────────────────────────────────────
    # Reward finding slots earlier in the episode
    components.efficiency = max(0.0, 1.0 - (step_count / max(max_steps, 1)))

    # ── 5. Soft Constraint Penalty ──────────────────────────────────────
    # Hidden traps: back-to-back, avoid days, fatigue, adversarial prefs
    penalty = 0.0
    day_name = dt.strftime("%A")

    for pid in participants:
        prof = profiles.get(pid)
        if not prof:
            continue

        avoid_days = prof.get("avoid_days") or [] if isinstance(prof, dict) else (getattr(prof, "avoid_days") or [])
        if day_name in avoid_days:
            penalty -= 0.4

        max_daily = prof.get("max_meetings_per_day", 99) if isinstance(prof, dict) else (getattr(prof, "max_meetings_per_day") or 99)
        fatigue = prof.get("fatigue_penalty", 0.0) if isinstance(prof, dict) else (getattr(prof, "fatigue_penalty") or 0.0)
        today_key = dt.date().isoformat()
        todays = [
            m for m in (participant_schedules.get(pid) or [])
            if (m["start"].date().isoformat() if hasattr(m["start"], "date") else str(m["start"])[:10]) == today_key
        ]
        if len(todays) >= max_daily:
            penalty -= fatigue

        soft = prof.get("soft_constraints") or {} if isinstance(prof, dict) else (getattr(prof, "soft_constraints") or {})
        if participant_schedules.get(pid):
            last = participant_schedules[pid][-1]
            gap_minutes = (dt - last["end"]).total_seconds() / 60 if hasattr(last["end"], "total_seconds") or hasattr(last["end"], "isoformat") else 999
            if isinstance(gap_minutes, (int, float)) and gap_minutes < 30:
                penalty += soft.get("back_to_back", 0)

        # Adversarial hidden preferences
        if isinstance(prof, dict) and "adversarial_constraints" in prof:
            for ac in prof["adversarial_constraints"]:
                if ac.get("type") == "no_meetings_during" and ac.get("hour_start", 0) <= dt.hour < ac.get("hour_end", 24):
                    penalty -= ac.get("penalty", 0.3)

    components.soft_constraint_penalty = max(-1.0, penalty)

    # ── Weighted Total ──────────────────────────────────────────────────
    components.total = round(
        REWARD_WEIGHTS["conflict_free"] * components.conflict_free
        + REWARD_WEIGHTS["preference_alignment"] * components.preference_alignment
        + REWARD_WEIGHTS["timezone_fairness"] * components.timezone_fairness
        + REWARD_WEIGHTS["efficiency"] * components.efficiency
        + REWARD_WEIGHTS["soft_constraint_penalty"] * abs(components.soft_constraint_penalty)
            * (-1 if components.soft_constraint_penalty < 0 else 1),
        4,
    )

    return components.total, components

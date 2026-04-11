"""
SchedulrX Environment
=====================
A POMDP-based multi-participant meeting scheduling environment for RL research.

Key design properties:
- Partial observability: participant preferences hidden until explicitly queried
- Active information gathering: agents call read_profile before scheduling
- Hard constraints: avoid_days enforced in slot validation (must discover via profile)
- Soft constraints: fatigue/back-to-back penalise but don't block
- Dense reward shaping throughout episode
- 3 difficulty tiers with genuine scaling
"""

import os
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pytz

from models.schemas import Action, HiddenProfile, MeetingRequest, Observation, Participant

def set_seed(seed=42):
    """Lock determinism across random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class SchedulrXEnv:
    def __init__(self):
        set_seed(42)
        self.current_task: Optional[str] = None
        self.done: bool = False
        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.max_steps: int = 80
        self.profiles: Dict[str, HiddenProfile] = {}
        self.scheduled: List[Dict] = []
        self.requests: List[MeetingRequest] = []
        self.participants: Dict[str, Participant] = {}
        self.profiles_read: Dict[str, HiddenProfile] = {}
        self.participant_schedules: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------ lifecycle

    def reset(self, task_name: str = "easy", seed: int = None) -> Observation:
        set_seed(seed or 42)
        self.current_task = task_name
        self.read_budget = {"easy": 2, "medium": 4, "hard": 5}[task_name]
        self.total_reads = 0
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.scheduled = []
        self.profiles_read = {}
        self.participant_schedules = {}
        self.trust_scores = {}
        self.adversarial_participant = None

        self.participants = {
            "p1": Participant(id="p1", name="Alice", timezone="Asia/Kolkata",    availability=[]),
            "p2": Participant(id="p2", name="Bob",   timezone="UTC",              availability=[]),
            "p3": Participant(id="p3", name="Carol", timezone="America/New_York", availability=[]),
            "p4": Participant(id="p4", name="Dave",  timezone="Europe/London",    availability=[]),
            "p5": Participant(id="p5", name="Eve",   timezone="Asia/Tokyo",       availability=[]),
        }

        # Hidden profiles — revealed ONLY via read_profile action.
        # avoid_days is a HARD scheduling constraint: any attempt to schedule
        # a participant on their avoid_day fails validation. Agents that skip
        # profile discovery will hit these walls and score poorly.
        self.profiles = {
            "p1": HiddenProfile(
                preferred_times=["morning"],
                avoid_days=["Saturday", "Sunday"],
                max_meetings_per_day=2,
                fatigue_penalty=0.30,
                soft_constraints={"back_to_back": -0.35},
            ),
            "p2": HiddenProfile(
                preferred_times=["afternoon"],
                avoid_days=[],
                max_meetings_per_day=3,
                fatigue_penalty=0.10,
                soft_constraints={"late_night": -0.50},
            ),
            "p3": HiddenProfile(
                preferred_times=["morning"],
                avoid_days=["Friday"],
                max_meetings_per_day=2,
                fatigue_penalty=0.40,
                soft_constraints={"monday": -0.60},
            ),
            "p4": HiddenProfile(
                preferred_times=["afternoon"],
                avoid_days=[],
                max_meetings_per_day=4,
                fatigue_penalty=0.15,
                soft_constraints={},
            ),
            "p5": HiddenProfile(
                preferred_times=["evening"],
                avoid_days=["Sunday", "Monday"],
                max_meetings_per_day=2,
                fatigue_penalty=0.35,
                soft_constraints={"back_to_back": -0.45},
            ),
        }

        for pid in self.participants:
            self.participant_schedules[pid] = []

        if task_name == "easy":
            # No avoid_day traps — good baseline for agent validation
            self.requests = [
                MeetingRequest(
                    id="r1", title="Team sync",
                    duration_minutes=45, priority=8,
                    participants=["p2", "p4"],  # p2,p4 have no avoid_days
                ),
            ]
        elif task_name == "medium":
            # One hidden trap: Carol (p3) avoids Friday.
            # The 7-day window always includes a Friday, so greedy agents fail r1.
            self.requests = [
                MeetingRequest(
                    id="r1", title="Project kickoff",
                    duration_minutes=60, priority=9,
                    participants=["p1", "p2", "p3"],
                ),
                MeetingRequest(
                    id="r2", title="Design review",
                    duration_minutes=60, priority=7,
                    participants=["p2", "p4"],
                ),
                MeetingRequest(
                    id="r3", title="Budget check",
                    duration_minutes=45, priority=6,
                    participants=["p1", "p5"],
                ),
            ]
        else:  # hard
            # Multiple traps: Alice avoids Sat/Sun, Eve avoids Sun/Mon, Carol avoids Fri.
            # Greedy agent will randomly pick days and fail repeatedly.
            # Optimal strategy: read p1, p3, p5 profiles first, then pick safe days.
            self.requests = [
                MeetingRequest(
                    id="r1", title="Strategy offsite",
                    duration_minutes=90, priority=10,
                    participants=["p1", "p2", "p3", "p4"],
                ),
                MeetingRequest(
                    id="r2", title="Investor call",
                    duration_minutes=60, priority=9,
                    participants=["p1", "p3", "p5"],
                ),
                MeetingRequest(
                    id="r3", title="Team retro",
                    duration_minutes=45, priority=5,
                    participants=["p2", "p4"],
                ),
            ]
            self.adversarial_participant = "p5"

        self.trust_scores = {p_id: 3 for p_id in self.participants}

        self._generate_availability()
        return self._get_observation()

    # ------------------------------------------------------------------ availability

    def _generate_availability(self):
        random.seed({"easy": 42, "medium": 137, "hard": 999}[self.current_task])
        now = datetime(2026, 4, 6, 9, 0, tzinfo=pytz.UTC)
        slot_hours = 3 if self.current_task == "hard" else 1

        for p in self.participants.values():
            tz = pytz.timezone(p.timezone)
            avail = []
            for day in range(5):
                start = now + timedelta(days=day)
                for hour in range(8, 18, slot_hours + 1):
                    if hour + slot_hours > 18:
                        break
                    slot_start = start.replace(hour=hour, minute=0).astimezone(tz).isoformat()
                    slot_end = (start.replace(hour=hour + slot_hours, minute=0)).astimezone(tz).isoformat()
                    avail.append({"start": slot_start, "end": slot_end})
            p.availability = avail

    def _overlaps(self, slot: Dict[str, str], meeting_start: datetime, meeting_end: datetime) -> bool:
        slot_start = datetime.fromisoformat(slot["start"].replace("Z", "+00:00"))
        slot_end = datetime.fromisoformat(slot["end"].replace("Z", "+00:00"))
        # overlap if one starts before the other ends
        return slot_start < meeting_end and slot_end > meeting_start

    # ------------------------------------------------------------------ step

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        info: Dict = {}
        if action.action_type == "read_profile" and action.participant_id:
            pid = action.participant_id
            if self.total_reads >= self.read_budget:
                reward = -0.15
                info["error"] = "Read budget exhausted"
            elif pid in self.profiles and pid not in self.profiles_read:
                self.profiles_read[pid] = self.profiles[pid]
                self.total_reads += 1
                reward = 0.2
                info["discovered"] = pid
            else:
                reward = -0.1  # already read or invalid pid

        elif (
            action.action_type == "schedule_meeting"
            and action.proposed_time
            and action.meeting_id
        ):
            if any(m["meeting_id"] == action.meeting_id for m in self.scheduled):
                reward = -0.5
                info["error"] = "already_scheduled"
                return self._get_observation(), reward, self.done, info

            req = next((r for r in self.requests if r.id == action.meeting_id), None)
            if not req:
                reward = -0.4
                info["error"] = "unknown_meeting_id"
                return self._get_observation(), reward, self.done, info

            # Adversarial participant: p5 in hard mode rejects 60% of
            # blind schedules (profile not read). Forces information gathering.
            if (
                self.adversarial_participant
                and self.adversarial_participant in req.participants
                and self.adversarial_participant not in self.profiles_read
            ):
                if random.random() < 0.6:
                    reward = -0.30
                    info["error"] = "adversarial_rejection"
                    info["cancelled"] = True
                    info["reason"] = f"{self.adversarial_participant} declined — hidden conflict"
                    return self._get_observation(), reward, self.done, info

            valid, constraint_delta, reason = self._validate_slot(action.proposed_time, req)
            if valid:
                self.scheduled.append({
                    "meeting_id": action.meeting_id,
                    "time": action.proposed_time,
                    "participants": req.participants,
                })
                self._update_participant_schedules(action.proposed_time, req)
                reward = 0.5 + constraint_delta
                info["scheduled"] = action.meeting_id

                # --- Cascading Availability ---
                # Strip overlapping slots from all participants in this meeting,
                # making early scheduling decisions irreversible.
                try:
                    sched_start = datetime.fromisoformat(
                        action.proposed_time.replace("Z", "+00:00")
                    )
                    sched_end = sched_start + timedelta(minutes=req.duration_minutes)
                    for pid in req.participants:
                        p = self.participants[pid]
                        if p.availability:
                            p.availability = [
                                slot for slot in p.availability
                                if not self._overlaps(slot, sched_start, sched_end)
                            ]
                except Exception:
                    pass  # never crash the env on cascade
            else:
                reward = -0.6
                info["error"] = reason

            # Dense progress signal — shaped throughout episode
            progress = len(self.scheduled) / max(len(self.requests), 1)
            reward += progress * 0.2
            reward = max(-1.0, min(1.0, reward))

        elif action.action_type == "propose_alternative":
            # Exploration-safe action — no penalty for proposing alternatives
            info["noted"] = action.meeting_id
            reward = 0.05

        self.total_reward += reward
        self.done = (
            self.step_count >= self.max_steps
            or len(self.scheduled) == len(self.requests)
        )
        return self._get_observation(), reward, self.done, info

    # ------------------------------------------------------------------ validation

    def _validate_slot(
        self, proposed_time: str, req: MeetingRequest
    ) -> Tuple[bool, float, str]:
        """
        Returns (is_valid, constraint_delta, failure_reason).

        Hard constraints (fail immediately):
          - Proposed time outside availability block for any participant
          - Proposed day is in participant's avoid_days

        Soft constraints (shift constraint_delta, don't block):
          - Back-to-back meetings < 30 min gap
          - Exceeding max_meetings_per_day
        """
        try:
            dt     = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
            end_dt = dt + timedelta(minutes=req.duration_minutes)
        except (ValueError, KeyError, TypeError):
            return False, 0.0, "invalid_datetime"

        constraint_delta = 0.0

        for pid in req.participants:
            p  = self.participants[pid]
            tz = pytz.timezone(p.timezone)
            local_dt  = dt.astimezone(tz)
            local_end = end_dt.astimezone(tz)
            local_day = DAY_NAMES[local_dt.weekday()]

            # Hard: avoid_days (agent must discover this via read_profile first)
            if pid in self.profiles and local_day in self.profiles[pid].avoid_days:
                return False, 0.0, f"{pid}_avoids_{local_day}"

            # Hard: availability block coverage
            in_block = any(
                local_dt  >= datetime.fromisoformat(slot["start"])
                and local_end <= datetime.fromisoformat(slot["end"])
                for slot in p.availability
            )
            if not in_block:
                return False, 0.0, f"{pid}_outside_availability"

            # Soft: only apply if profile has been read
            if pid in self.profiles_read:
                profile = self.profiles[pid]
                prev = self.participant_schedules.get(pid, [])
                if prev:
                    gap = (dt - prev[-1]["end"]).total_seconds() / 60
                    if gap < 30:
                        constraint_delta += profile.soft_constraints.get("back_to_back", 0.0)
                today_count = sum(
                    1 for m in prev if m["start"].date() == local_dt.date()
                )
                if today_count >= profile.max_meetings_per_day:
                    constraint_delta -= profile.fatigue_penalty

        return True, constraint_delta, ""

    def _update_participant_schedules(self, proposed_time: str, req: MeetingRequest):
        dt     = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        end_dt = dt + timedelta(minutes=req.duration_minutes)
        for pid in req.participants:
            self.participant_schedules.setdefault(pid, []).append(
                {"start": dt, "end": end_dt, "meeting_id": req.id}
            )

    # ------------------------------------------------------------------ observation / state

    def _get_observation(self) -> Observation:
        return Observation(
            current_time=datetime.now(pytz.UTC),
            participants=list(self.participants.values()),
            requests=self.requests,
            scheduled_meetings=self.scheduled,
            profiles_read=self.profiles_read,
            step_count=self.step_count,
            trust_scores=self.trust_scores,
        )

    def state(self) -> Dict:
        scheduled_ids = {m["meeting_id"] for m in self.scheduled}
        return {
            "task": self.current_task,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "total_reward": round(self.total_reward, 4),
            "scheduled_count": len(self.scheduled),
            "total_requests": len(self.requests),
            "pending_meetings": [r.id for r in self.requests if r.id not in scheduled_ids],
            "profiles_read": list(self.profiles_read.keys()),
            "read_budget_remaining": self.read_budget - self.total_reads,
            "scheduled_meetings": self.scheduled,
        }

    # ------------------------------------------------------------------ grader

    def get_grader_score(self) -> Dict:
        completed = len(self.scheduled)
        total = len(self.requests)

        if total == 0 or completed == 0:
            return {"score": 0.0, "completed": 0, "violations": 0, "task": self.current_task}

        base = completed / total

        violations = sum(
            0.3 for m in self.scheduled
            for pid in m["participants"]
            if pid in self.profiles and pid not in self.profiles_read
        )
        constraint_score = max(0.0, 1.0 - violations)
        step_efficiency = max(0.0, 1.0 - (self.step_count / self.max_steps))

        if self.current_task == "easy":
            final_score = min(1.0, base * 0.85 + step_efficiency * 0.15)
        elif self.current_task == "medium":
            final_score = min(1.0, base * 0.55 + constraint_score * 0.35 + step_efficiency * 0.10)
        else:  # hard
            adversarial_read = 1.0 if "p5" in self.profiles_read else 0.0
            final_score = min(1.0, base * 0.40 + constraint_score * 0.35 + adversarial_read * 0.15 + step_efficiency * 0.10)

        return {
            "score": round(final_score, 3),
            "completed": completed,
            "total": total,
            "violations": round(violations, 2),
            "task": self.current_task
        }

def programmatic_grade(env: SchedulrXEnv = None, **kwargs) -> Dict:
    """Entry point for programmatic evaluators."""
    if env is None:
        env = SchedulrXEnv()
        env.reset(kwargs.get("task_name", "easy"))
    return env.get_grader_score()

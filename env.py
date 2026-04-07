import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import pytz
from models.schemas import Observation, Action, Participant, MeetingRequest, HiddenProfile


class SchedulrXEnv:
    def __init__(self):
        self.current_task = None
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.max_steps = 30
        self.profiles: Dict[str, HiddenProfile] = {}
        self.scheduled: List[Dict] = []
        self.requests: List[MeetingRequest] = []
        self.participants: Dict[str, Participant] = {}
        self.profiles_read: Dict[str, HiddenProfile] = {}
        self.participant_schedules: Dict[str, List[Dict]] = {}
        self._rng = random.Random()

    def reset(self, task_name: str = "easy", seed: int = 42) -> "Observation":
        self._rng.seed(seed)
        self.current_task = task_name
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.scheduled = []
        self.profiles_read = {}
        self.participant_schedules = {}

        self.participants = {
            "p1": Participant(id="p1", name="Alice", timezone="Asia/Kolkata",    availability=[]),
            "p2": Participant(id="p2", name="Bob",   timezone="UTC",             availability=[]),
            "p3": Participant(id="p3", name="Carol", timezone="America/New_York", availability=[]),
            "p4": Participant(id="p4", name="Dave",  timezone="Europe/London",   availability=[]),
            "p5": Participant(id="p5", name="Eve",   timezone="Asia/Tokyo",      availability=[]),
        }

        self.profiles = {
            "p1": HiddenProfile(preferred_times=["morning"],   avoid_days=["Saturday"],
                                max_meetings_per_day=2, fatigue_penalty=0.3,
                                soft_constraints={"back_to_back": -0.4}),
            "p2": HiddenProfile(preferred_times=["afternoon"], avoid_days=[],
                                max_meetings_per_day=3, fatigue_penalty=0.1,
                                soft_constraints={"late_night": -0.6}),
            "p3": HiddenProfile(preferred_times=["morning"],   avoid_days=["Friday"],
                                max_meetings_per_day=1, fatigue_penalty=0.5,
                                soft_constraints={"monday": -0.3}),
            "p4": HiddenProfile(preferred_times=["afternoon"], avoid_days=[],
                                max_meetings_per_day=4, fatigue_penalty=0.2,
                                soft_constraints={}),
            "p5": HiddenProfile(preferred_times=["evening"],   avoid_days=["Sunday"],
                                max_meetings_per_day=2, fatigue_penalty=0.4,
                                soft_constraints={"back_to_back": -0.5}),
        }

        for pid in self.participants:
            self.participant_schedules[pid] = []

        if task_name == "easy":
            self.requests = [
                MeetingRequest(id="r1", title="Team sync",
                               duration_minutes=30, priority=8,
                               participants=["p1", "p2"]),
            ]
        elif task_name == "medium":
            self.requests = [
                MeetingRequest(id="r1", title="Project kickoff",
                               duration_minutes=45, priority=9,
                               participants=["p1", "p2", "p3"]),
                MeetingRequest(id="r2", title="Design review",
                               duration_minutes=60, priority=7,
                               participants=["p2", "p4"]),
                MeetingRequest(id="r3", title="Budget check",
                               duration_minutes=30, priority=6,
                               participants=["p1", "p5"]),
            ]
        else:  # hard — FIX: was 90 min (impossible), now 60 min
            self.requests = [
                MeetingRequest(id="r1", title="Strategy offsite",
                               duration_minutes=60, priority=10,
                               participants=["p1", "p2", "p3", "p4"]),
                MeetingRequest(id="r2", title="Investor call",
                               duration_minutes=60, priority=9,
                               participants=["p1", "p3", "p5"]),
                MeetingRequest(id="r3", title="Team retro",
                               duration_minutes=45, priority=5,
                               participants=["p2", "p4"]),
            ]

        self._generate_availability()
        return self._get_observation()

    def _generate_availability(self):
        # FIX: dynamic date, not hardcoded April 6
        now = datetime.now(pytz.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        for p in self.participants.values():
            tz = pytz.timezone(p.timezone)
            avail = []
            profile = self.profiles.get(p.id)
            for day in range(5):
                day_start = now + timedelta(days=day)
                day_name  = day_start.strftime("%A")
                if profile and day_name in profile.avoid_days:
                    continue
                for hour in range(9, 17, 2):          # 9,11,13,15 — 2-hr slots
                    slot_start = day_start.replace(hour=hour,     minute=0)
                    slot_end   = day_start.replace(hour=hour + 2, minute=0)
                    avail.append({
                        "start": slot_start.astimezone(tz).isoformat(),
                        "end":   slot_end.astimezone(tz).isoformat(),
                    })
            p.availability = avail

    def step(self, action: Action) -> Tuple["Observation", float, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        info   = {}

        if action.action_type == "read_profile" and action.participant_id:
            pid = action.participant_id
            if pid in self.profiles and pid not in self.profiles_read:
                self.profiles_read[pid] = self.profiles[pid]
                reward += 0.25
                info["discovered"] = pid
            else:
                reward -= 0.1

        elif action.action_type == "schedule_meeting" \
                and action.proposed_time and action.meeting_id:

            if any(m["meeting_id"] == action.meeting_id for m in self.scheduled):
                reward -= 0.8
                return self._get_observation(), reward, self.done, info

            req = next((r for r in self.requests if r.id == action.meeting_id), None)
            if not req:
                reward -= 0.6
                return self._get_observation(), reward, self.done, info

            valid, constraint_delta = self._validate_slot(action.proposed_time, req)
            if valid:
                self.scheduled.append({
                    "meeting_id":  action.meeting_id,
                    "time":        action.proposed_time,
                    "participants": req.participants,
                })
                self._update_participant_schedules(action.proposed_time, req)
                reward += 0.45 + constraint_delta
            else:
                reward -= 0.7
        else:
            reward -= 0.05
            info["error"] = f"unknown action_type: {action.action_type}"

        progress = min(len(self.scheduled) / max(len(self.requests), 1), 1.0)
        reward  += progress * 0.25
        reward   = max(-1.0, min(1.0, reward))
        self.total_reward += reward

        self.done = (
            self.step_count >= self.max_steps
            or len(self.scheduled) == len(self.requests)
        )
        return self._get_observation(), reward, self.done, info

    def _validate_slot(self, proposed_time: str, req: MeetingRequest) -> Tuple[bool, float]:
        try:
            dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return False, 0.0
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)

        end_dt            = dt + timedelta(minutes=req.duration_minutes)
        constraint_delta  = 0.0
        day_name          = dt.strftime("%A")

        for pid in req.participants:
            p  = self.participants[pid]
            tz = pytz.timezone(p.timezone)
            local_dt  = dt.astimezone(tz)
            local_end = end_dt.astimezone(tz)

            in_slot = False
            for slot in p.availability:
                try:
                    s_start = datetime.fromisoformat(slot["start"])
                    s_end   = datetime.fromisoformat(slot["end"])
                    # FIX: ensure both are timezone-aware before comparing
                    if s_start.tzinfo is None:
                        s_start = pytz.utc.localize(s_start)
                    if s_end.tzinfo is None:
                        s_end = pytz.utc.localize(s_end)
                    if local_dt >= s_start and local_end <= s_end:
                        in_slot = True
                        break
                except (ValueError, KeyError):
                    continue

            if not in_slot:
                return False, 0.0

            profile = self.profiles.get(pid)
            if profile:
                if day_name in profile.avoid_days:
                    constraint_delta -= 0.35
                today_str   = dt.date().isoformat()
                # FIX: per-day count, not total
                todays = [
                    m for m in self.participant_schedules.get(pid, [])
                    if m["start"].date().isoformat() == today_str
                ]
                if len(todays) >= profile.max_meetings_per_day:
                    constraint_delta -= profile.fatigue_penalty
                if self.participant_schedules.get(pid):
                    last = self.participant_schedules[pid][-1]
                    gap  = (dt - last["end"]).total_seconds() / 60
                    if gap < 30:
                        constraint_delta += profile.soft_constraints.get("back_to_back", 0)

        return True, constraint_delta

    def _update_participant_schedules(self, proposed_time: str, req: MeetingRequest):
        dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        end_dt = dt + timedelta(minutes=req.duration_minutes)
        for pid in req.participants:
            self.participant_schedules.setdefault(pid, []).append({
                "start":      dt,
                "end":        end_dt,
                "meeting_id": req.id,
            })

    def _get_observation(self) -> "Observation":
        return Observation(
            current_time=datetime.now(pytz.UTC),
            participants=list(self.participants.values()),
            requests=self.requests,
            scheduled_meetings=self.scheduled,
            profiles_read=self.profiles_read,
            step_count=self.step_count,
        )

    def state(self) -> Dict:
        return {
            "task":          self.current_task,
            "step":          self.step_count,
            "max_steps":     self.max_steps,
            "scheduled":     len(self.scheduled),
            "total":         len(self.requests),
            "profiles_read": list(self.profiles_read.keys()),
            "done":          self.done,
        }

    def get_grader_score(self) -> Dict:
        completed = len(self.scheduled)
        total     = len(self.requests)
        base      = min(completed / total, 1.0) if total > 0 else 0.0

        violations = sum(
            0.3
            for m in self.scheduled
            for pid in m["participants"]
            if pid in self.profiles and pid not in self.profiles_read
        )
        constraint_score = max(0.0, 1.0 - violations)
        score = min(1.0, (base * 0.6) + (constraint_score * 0.4))
        return {
            "score":            round(score, 3),
            "completed":        completed,
            "total":            total,
            "violations":       round(violations, 2),
            "constraint_score": round(constraint_score, 3),
        }

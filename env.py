import random
from datetime import datetime, timedelta
import pytz
from typing import List, Tuple, Dict
from models.schemas import Observation, Action, Reward, Participant, MeetingRequest, HiddenProfile

TASK_SEEDS = {"easy": 42, "medium": 137, "hard": 999}

class SchedulrXEnv:
    def __init__(self):
        self.current_task = None
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.max_steps = 80
        self.profiles: Dict[str, HiddenProfile] = {}
        self.scheduled: List[Dict] = []
        self.requests: List[MeetingRequest] = []
        self.participants: Dict[str, Participant] = {}
        self.profiles_read: Dict[str, HiddenProfile] = {}
        self.participant_schedules: Dict[str, List[Dict]] = {}
        self.read_budget = 0
        self.total_reads = 0

    def reset(self, task_name: str = "easy") -> Observation:
        random.seed(TASK_SEEDS[task_name])
        self.current_task = task_name
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.scheduled = []
        self.profiles_read = {}
        self.participant_schedules = {}
        self.read_budget = {"easy": 2, "medium": 4, "hard": 5}[task_name]
        self.total_reads = 0

        self.participants = {
            "p1": Participant(id="p1", name="Alice", timezone="Asia/Kolkata", availability=[]),
            "p2": Participant(id="p2", name="Bob", timezone="UTC", availability=[]),
            "p3": Participant(id="p3", name="Carol", timezone="America/New_York", availability=[]),
            "p4": Participant(id="p4", name="Dave", timezone="Europe/London", availability=[]),
            "p5": Participant(id="p5", name="Eve", timezone="Asia/Tokyo", availability=[]),
        }

        self.profiles = {
            "p1": HiddenProfile(preferred_times=["morning"], avoid_days=["Saturday"], max_meetings_per_day=2, fatigue_penalty=0.3, soft_constraints={"back_to_back": -0.4}),
            "p2": HiddenProfile(preferred_times=["afternoon"], avoid_days=[], max_meetings_per_day=3, fatigue_penalty=0.1, soft_constraints={"late_night": -0.6}),
            "p3": HiddenProfile(preferred_times=["morning"], avoid_days=["Friday"], max_meetings_per_day=1, fatigue_penalty=0.5, soft_constraints={"monday": -0.7}),
            "p4": HiddenProfile(preferred_times=["afternoon"], avoid_days=[], max_meetings_per_day=4, fatigue_penalty=0.2, soft_constraints={}),
            "p5": HiddenProfile(preferred_times=["evening"], avoid_days=["Sunday"], max_meetings_per_day=2, fatigue_penalty=0.4, soft_constraints={"back_to_back": -0.5}),
        }

        for pid in self.participants:
            self.participant_schedules[pid] = []

        if task_name == "easy":
            self.requests = [
                MeetingRequest(id="r1", title="Team sync", duration_minutes=30, priority=8, participants=["p1", "p2"])
            ]
        elif task_name == "medium":
            self.requests = [
                MeetingRequest(id="r1", title="Project kickoff", duration_minutes=45, priority=9, participants=["p1", "p2", "p3"]),
                MeetingRequest(id="r2", title="Design review", duration_minutes=60, priority=7, participants=["p2", "p4"]),
                MeetingRequest(id="r3", title="Budget check", duration_minutes=30, priority=6, participants=["p1", "p5"]),
            ]
        else:  # hard
            self.requests = [
                MeetingRequest(id="r1", title="Strategy offsite", duration_minutes=90, priority=10, participants=["p1", "p2", "p3", "p4"]),
                MeetingRequest(id="r2", title="Investor call", duration_minutes=60, priority=9, participants=["p1", "p3", "p5"]),
                MeetingRequest(id="r3", title="Team retro", duration_minutes=45, priority=5, participants=["p2", "p4"]),
            ]

        self._generate_availability()
        return self._get_observation()

    def _generate_availability(self):
        now = datetime(2026, 4, 6, 9, 0, tzinfo=pytz.UTC)
        # Hard task needs 3-hour slots to fit the 90-minute Strategy offsite
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
                    slot_end = start.replace(hour=hour + slot_hours, minute=0).astimezone(tz).isoformat()
                    avail.append({"start": slot_start, "end": slot_end})
            p.availability = avail

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        info = {}

        if action.action_type == "read_profile" and action.participant_id:
            if self.total_reads >= self.read_budget:
                reward -= 0.15
                info["error"] = "Read budget exhausted"
            elif action.participant_id in self.profiles and action.participant_id not in self.profiles_read:
                self.profiles_read[action.participant_id] = self.profiles[action.participant_id]
                self.total_reads += 1
                reward += 0.2
                info["discovered"] = action.participant_id
            else:
                reward -= 0.1  # already read or invalid pid

        elif action.action_type == "schedule_meeting" and action.proposed_time and action.meeting_id:
            if any(m["meeting_id"] == action.meeting_id for m in self.scheduled):
                reward -= 0.8
                return self._get_observation(), reward, self.done, info

            req = next((r for r in self.requests if r.id == action.meeting_id), None)
            if not req:
                reward -= 0.6
                return self._get_observation(), reward, self.done, info

            valid, constraint_violation = self._validate_slot(action.proposed_time, req)
            if valid:
                self.scheduled.append({
                    "meeting_id": action.meeting_id,
                    "time": action.proposed_time,
                    "participants": req.participants
                })
                self._update_participant_schedules(action.proposed_time, req)
                reward += 0.45 + constraint_violation
                if self.current_task == "hard" and action.meeting_id == "r1" and "p5" not in self.profiles_read:
                    info["hint"] = "Strategy offsite scheduled blind — consider reading P5 profile"
            else:
                reward -= 0.7

        progress = min(len(self.scheduled) / max(len(self.requests), 1), 1.0)
        reward += progress * 0.25
        reward = max(-1.0, min(1.0, reward))
        self.total_reward += reward

        self.done = self.step_count >= self.max_steps or len(self.scheduled) == len(self.requests)
        return self._get_observation(), reward, self.done, info

    def _validate_slot(self, proposed_time: str, req: MeetingRequest) -> Tuple[bool, float]:
        try:
            dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
            duration = timedelta(minutes=req.duration_minutes)
            end_dt = dt + duration
        except (ValueError, KeyError, TypeError):
            return False, -1.0

        constraint_violation = 0.0
        for pid in req.participants:
            p = self.participants[pid]
            tz = pytz.timezone(p.timezone)
            local_dt = dt.astimezone(tz)
            local_end = end_dt.astimezone(tz)

            in_slot = any(
                local_dt >= datetime.fromisoformat(slot["start"]) and
                local_end <= datetime.fromisoformat(slot["end"])
                for slot in p.availability
            )
            if not in_slot:
                return False, -1.0

            if pid in self.profiles:
                profile = self.profiles[pid]
                if self.participant_schedules.get(pid):
                    last = self.participant_schedules[pid][-1]
                    gap_minutes = (dt - last["end"]).total_seconds() / 60
                    if gap_minutes < 30:
                        constraint_violation += profile.soft_constraints.get("back_to_back", 0)
                if len(self.participant_schedules.get(pid, [])) >= profile.max_meetings_per_day:
                    constraint_violation -= profile.fatigue_penalty

        return True, constraint_violation

    def _update_participant_schedules(self, proposed_time: str, req: MeetingRequest):
        dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        end_dt = dt + timedelta(minutes=req.duration_minutes)
        for pid in req.participants:
            if pid not in self.participant_schedules:
                self.participant_schedules[pid] = []
            self.participant_schedules[pid].append({"start": dt, "end": end_dt, "meeting_id": req.id})

    def _get_observation(self) -> Observation:
        return Observation(
            current_time=datetime(2026, 4, 6, 9, 0, tzinfo=pytz.UTC),  # fixed, not datetime.now()
            participants=list(self.participants.values()),
            requests=self.requests,
            scheduled_meetings=self.scheduled,
            profiles_read=self.profiles_read,
            step_count=self.step_count
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

    def get_grader_score(self) -> Dict:
        completed = len(self.scheduled)
        total = len(self.requests)

        if total == 0 or completed == 0:
            return {"score": 0.0, "completed": 0, "total": total, "violations": 0, "task": self.current_task}

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

import random
from datetime import datetime, timedelta
import pytz
from typing import List, Tuple, Dict, Optional
from models.schemas import Observation, Action, Reward, Participant, MeetingRequest, HiddenProfile

TASK_SEEDS = {"easy": 42, "medium": 137, "hard": 999}

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
        self.cancelled_meetings: List[str] = []
        self.counter_proposals: List[Dict] = []
        self.cancellation_step: Optional[int] = None
        self.read_budget = 0
        self.total_reads = 0

    def reset(self, task_name: str = "easy", seed: Optional[int] = None) -> Observation:
        set_seed = seed if seed is not None else TASK_SEEDS.get(task_name, 42)
        random.seed(set_seed)
        import numpy as np
        np.random.seed(set_seed)
        self.current_task = task_name
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.max_steps = {"easy": 30, "medium": 30, "hard": 40}.get(task_name, 30)
        self.scheduled = []
        self.profiles_read = {}
        self.participant_schedules = {}
        self.cancelled_meetings = []
        self.counter_proposals = []
        self.cancellation_step = random.randint(8, 14) if task_name == "hard" else None
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
                MeetingRequest(id="r2", title="Design review", duration_minutes=60, priority=7, participants=["p2", "p4"], depends_on="r1"),
                MeetingRequest(id="r3", title="Budget check", duration_minutes=30, priority=6, participants=["p1", "p5"]),
            ]
        else:  # hard
            self.requests = [
                MeetingRequest(id="r1", title="Strategy offsite", duration_minutes=90, priority=10, participants=["p1", "p2", "p3", "p4"]),
                MeetingRequest(id="r2", title="Investor call", duration_minutes=60, priority=9, participants=["p1", "p3", "p5"], depends_on="r1"),
                MeetingRequest(id="r3", title="Team retro", duration_minutes=45, priority=5, participants=["p2", "p4"]),
            ]

        self._generate_availability()
        return self._get_observation()

    def _generate_availability(self):
        now = datetime(2026, 4, 6, 9, 0, tzinfo=pytz.UTC)
        # Switch to continuous availability block (9 AM to 6 PM) to ensure solvability
        for p in self.participants.values():
            tz = pytz.timezone(p.timezone)
            avail = []
            for day in range(5):
                start = now + timedelta(days=day)
                # Create one large continuous block (9 AM to 6 PM) in the participant's LOCAL timezone
                local_start = tz.localize(start.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=None))
                local_end = tz.localize(start.replace(hour=18, minute=0, second=0, microsecond=0, tzinfo=None))
                avail.append({"start": local_start.isoformat(), "end": local_end.isoformat()})
            p.availability = avail

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        info = {}
        last_event = None

        # Stochastic cancellation mechanic in Hard Mode
        if self.current_task == "hard" and self.step_count == self.cancellation_step and self.scheduled:
            # Cancel a random meeting
            cancelled = random.choice(self.scheduled)
            self.scheduled = [m for m in self.scheduled if m["meeting_id"] != cancelled["meeting_id"]]
            self.cancelled_meetings.append(cancelled["meeting_id"])
            # Remove from participant schedules
            for pid in cancelled["participants"]:
                self.participant_schedules[pid] = [
                    m_sch for m_sch in self.participant_schedules.get(pid, [])
                    if m_sch["meeting_id"] != cancelled["meeting_id"]
                ]
            last_event = f"CRITICAL: Meeting {cancelled['meeting_id']} was unexpectedly cancelled by participants. You must reschedule it."

        if action.action_type == "read_profile" and action.participant_id:
            if self.total_reads >= self.read_budget:
                reward -= 0.15
                info["error"] = "Read budget exhausted"
            elif action.participant_id in self.profiles and action.participant_id not in self.profiles_read:
                self.profiles_read[action.participant_id] = self.profiles[action.participant_id]
                self.total_reads += 1
                reward += 0.0
                info["discovered"] = action.participant_id
            else:
                reward -= 0.1  # already read or invalid pid

        elif action.action_type in ("schedule_meeting", "reschedule_meeting") and action.proposed_time and action.meeting_id:
            if any(m["meeting_id"] == action.meeting_id for m in self.scheduled):
                reward -= 0.8
                return self._get_observation(last_event), reward, self.done, info

            if action.action_type == "reschedule_meeting" and action.meeting_id not in self.cancelled_meetings:
                reward -= 0.5
                return self._get_observation(last_event), reward, self.done, info

            req = next((r for r in self.requests if r.id == action.meeting_id), None)
            if not req:
                reward -= 0.6
                return self._get_observation(last_event), reward, self.done, info

            # Check Cascading Dependencies
            if hasattr(req, "depends_on") and req.depends_on is not None:
                if not any(m["meeting_id"] == req.depends_on for m in self.scheduled):
                    reward -= 0.5
                    last_event = f"Dependency unmet: Cannot schedule {req.id} before {req.depends_on}."
                    return self._get_observation(last_event), reward, self.done, info

            valid, constraint_violation, reason = self._validate_slot(action.proposed_time, req)
            if valid:
                # Remove from cancelled list if rescheduling
                if action.action_type == "reschedule_meeting":
                    self.cancelled_meetings.remove(action.meeting_id)
                    reward += 0.50
                else:
                    reward += 0.45

                self.scheduled.append({
                    "meeting_id": action.meeting_id,
                    "time": action.proposed_time,
                    "participants": req.participants
                })
                self._update_participant_schedules(action.proposed_time, req)
                reward += constraint_violation
                
                if self.current_task == "hard" and action.meeting_id == "r1" and "p5" not in self.profiles_read:
                    info["hint"] = "Strategy offsite scheduled blind — consider reading P5 profile"
            else:
                # Issue Counter-Proposal Randomly on Soft Failures
                if reason == "out_of_slot" and random.random() < 0.5:
                    prop_id = f"cp_{len(self.counter_proposals)}"
                    # Naively propose 2 hours later
                    try:
                        dt = datetime.fromisoformat(action.proposed_time.replace("Z", "+00:00")) + timedelta(hours=2)
                        self.counter_proposals.append({
                            "proposal_id": prop_id,
                            "meeting_id": action.meeting_id,
                            "proposed_time": dt.isoformat(),
                            "reason": "Participant proposed alternative time due to conflict."
                        })
                        last_event = f"Proposal REJECTED. Counter-proposal generated: {prop_id}."
                        reward -= 0.25 # Softer penalty for triggering a counter-proposal
                    except (ValueError, TypeError, KeyError):
                        reward -= 0.7
                else:
                    reward -= 0.7
                    last_event = f"Proposal REJECTED: {reason}"

        elif action.action_type == "accept_proposal" and action.proposal_id:
            prop = next((p for p in self.counter_proposals if p["proposal_id"] == action.proposal_id), None)
            if prop:
                req = next((r for r in self.requests if r.id == prop["meeting_id"]), None)
                if req:
                    self.counter_proposals.remove(prop)
                    # For simplicity in testing, accept_proposal explicitly overrides minor validation failures.
                    if prop["meeting_id"] in self.cancelled_meetings: self.cancelled_meetings.remove(prop["meeting_id"])
                    self.scheduled.append({
                        "meeting_id": req.id,
                        "time": prop["proposed_time"],
                        "participants": req.participants
                    })
                    self._update_participant_schedules(prop["proposed_time"], req)
                    reward += 0.55
                    last_event = f"Accepted counter-proposal {action.proposal_id} for {req.id}."
            else:
                reward -= 0.05
        else:
            if not action.action_type == "read_profile":
                reward -= 0.05

        progress = min(len(self.scheduled) / max(len(self.requests), 1), 1.0)
        reward += progress * 0.20 # progress bonus mapping to yaml
        reward = max(-1.0, min(1.0, reward))
        self.total_reward += reward

        self.done = self.step_count >= self.max_steps or len(self.scheduled) == len(self.requests)
        return self._get_observation(last_event), reward, self.done, info

    def _validate_slot(self, proposed_time: str, req: MeetingRequest) -> Tuple[bool, float, str]:
        try:
            dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
            duration = timedelta(minutes=req.duration_minutes)
            end_dt = dt + duration
        except (ValueError, KeyError, TypeError):
            return False, -1.0, "invalid_time_format"

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
                return False, -1.0, "out_of_slot"

            # Adversarial Hard Requirement: P5 is the primary holder of the offsite venue
            if self.current_task == "hard" and req.id == "r1" and "p5" not in self.profiles_read:
                return False, -0.9, "adversarial_rejection"

            if pid in self.profiles:
                profile = self.profiles[pid]
                if self.participant_schedules.get(pid):
                    last = self.participant_schedules[pid][-1]
                    gap_minutes = (dt - last["end"]).total_seconds() / 60
                    if gap_minutes < 30:
                        constraint_violation += profile.soft_constraints.get("back_to_back", 0)
                if len(self.participant_schedules.get(pid, [])) >= profile.max_meetings_per_day:
                    constraint_violation -= profile.fatigue_penalty

        return True, constraint_violation, "ok"

    def _update_participant_schedules(self, proposed_time: str, req: MeetingRequest):
        dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        end_dt = dt + timedelta(minutes=req.duration_minutes)
        for pid in req.participants:
            if pid not in self.participant_schedules:
                self.participant_schedules[pid] = []
            self.participant_schedules[pid].append({"start": dt, "end": end_dt, "meeting_id": req.id})

    def _get_observation(self, last_event: str = None) -> Observation:
        return Observation(
            current_time=datetime(2026, 4, 6, 9, 0, tzinfo=pytz.UTC),
            participants=list(self.participants.values()),
            requests=self.requests,
            scheduled_meetings=self.scheduled,
            cancelled_meetings=self.cancelled_meetings,
            profiles_read=self.profiles_read,
            counter_proposals=self.counter_proposals,
            step_count=self.step_count,
            trust_scores={p_id: max(self.read_budget - self.total_reads, 0) for p_id in self.participants.keys()},
            last_event=last_event
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
            "cancelled_meetings": self.cancelled_meetings,
            "counter_proposals": self.counter_proposals
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
        constraint_score = max(0.0, base - violations)
        step_efficiency = max(0.0, 1.0 - (self.step_count / self.max_steps))

        if self.current_task == "easy":
            final_score = min(1.0, base * 0.90 + step_efficiency * 0.10)
        elif self.current_task == "medium":
            final_score = min(1.0, base * 0.70 + constraint_score * 0.25 + step_efficiency * 0.05)
        else:  # hard: requires high effort (completion) and adversarial discovery (p5)
            adversarial_read = 1.0 if "p5" in self.profiles_read else 0.0
            final_score = min(1.0, base * 0.60 + constraint_score * 0.25 + adversarial_read * 0.10 + step_efficiency * 0.05)

        return {
            "score": round(final_score, 3),
            "completed": completed,
            "total": total,
            "violations": round(violations, 2),
            "task": self.current_task
        }

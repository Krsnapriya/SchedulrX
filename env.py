import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import pytz
from models.schemas import Observation, Action, Participant, MeetingRequest, HiddenProfile


import uuid

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
        self.counter_proposals: List[Dict] = []
        self.cancelled_meetings: List[str] = []
        self.episode_start_time = None

    def reset(self, task_name: str = "easy", seed: int = 42) -> "Observation":
        self._rng.seed(seed)
        self.current_task = task_name
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.scheduled = []
        self.profiles_read = {}
        self.participant_schedules = {}
        self.counter_proposals = []
        self.cancelled_meetings = []
        
        # episode starts at "now" (rounded to start of next day)
        self.episode_start_time = datetime.now(pytz.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        self.participants = {
            "p1": Participant(id="p1", name="Alice", timezone="Asia/Kolkata",    availability=None),
            "p2": Participant(id="p2", name="Bob",   timezone="UTC",             availability=None),
            "p3": Participant(id="p3", name="Carol", timezone="America/New_York", availability=None),
            "p4": Participant(id="p4", name="Dave",  timezone="Europe/London",   availability=None),
            "p5": Participant(id="p5", name="Eve",   timezone="Asia/Tokyo",      availability=None),
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
                               participants=["p2", "p4"], depends_on="r1"),
                MeetingRequest(id="r3", title="Budget check",
                               duration_minutes=30, priority=6,
                               participants=["p1", "p5"]),
            ]
        else:  # hard
            self.requests = [
                MeetingRequest(id="r1", title="Strategy offsite",
                               duration_minutes=60, priority=10,
                               participants=["p1", "p2", "p3", "p4"],
                               deadline_hours=24), # Must be scheduled on the first day
                MeetingRequest(id="r2", title="Investor call",
                               duration_minutes=60, priority=9,
                               participants=["p1", "p3", "p5"], depends_on="r1"),
                MeetingRequest(id="r3", title="Team retro",
                               duration_minutes=45, priority=5,
                               participants=["p2", "p4"]),
            ]

        self._generate_availability()
        return self._get_observation()

    def _generate_availability(self):
        """God-tier stochastic availability: non-uniform slots."""
        for p in self.participants.values():
            tz = pytz.timezone(p.timezone)
            avail = []
            profile = self.profiles.get(p.id)
            for day in range(5):
                day_start = self.episode_start_time + timedelta(days=day)
                day_name  = day_start.strftime("%A")
                if profile and day_name in profile.avoid_days:
                    continue
                
                # Generate 2-3 random slots per day
                num_slots = self._rng.randint(2, 3)
                last_end_hour = 9
                for _ in range(num_slots):
                    if last_end_hour >= 14: break
                    start_hour = self._rng.randint(last_end_hour, 14)
                    duration = self._rng.randint(60, 180) # 1 to 3 hours
                    
                    slot_start = day_start.replace(hour=start_hour, minute=0)
                    slot_end = slot_start + timedelta(minutes=duration)
                    
                    avail.append({
                        "start": slot_start.astimezone(tz).isoformat(),
                        "end":   slot_end.astimezone(tz).isoformat(),
                    })
                    last_end_hour = slot_end.hour + 1
                    if last_end_hour >= 17: break

            p.availability = avail

    def step(self, action: Action) -> Tuple["Observation", float, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        info   = {}
        
        # --- Stochastic Cancellation (Hard task only) ---
        if self.current_task == "hard" and self.step_count == 12 and self.scheduled:
            victim = self._rng.choice(self.scheduled)
            self.scheduled.remove(victim)
            self.cancelled_meetings.append(victim["meeting_id"])
            info["cancellation"] = victim["meeting_id"]

        if action.action_type == "read_profile" and action.participant_id:
            pid = action.participant_id
            if pid in self.profiles and pid not in self.profiles_read:
                self.profiles_read[pid] = self.profiles[pid]
                reward += 0.25
                info["discovered"] = pid
            else:
                reward -= 0.1

        elif action.action_type == "schedule_meeting" or action.action_type == "reschedule_meeting":
            mid = action.meeting_id
            m_time = action.proposed_time
            
            if not mid or not m_time:
                reward -= 0.5
                return self._get_observation(), reward, self.done, info

            # If reschedule, remove old one first
            if action.action_type == "reschedule_meeting":
                self.scheduled = [m for m in self.scheduled if m["meeting_id"] != mid]
            elif any(m["meeting_id"] == mid for m in self.scheduled):
                reward -= 0.8 # duplication error
                return self._get_observation(), reward, self.done, info

            req = next((r for r in self.requests if r.id == mid), None)
            if not req:
                reward -= 0.6
                return self._get_observation(), reward, self.done, info

            valid, constraint_delta = self._validate_slot(m_time, req)
            if valid:
                self.scheduled.append({
                    "meeting_id":  mid,
                    "time":        m_time,
                    "participants": req.participants,
                })
                self._update_participant_schedules(m_time, req)
                reward += 0.5 + constraint_delta
                # Remove any existing counter-proposals for this meeting
                self.counter_proposals = [cp for cp in self.counter_proposals if cp["meeting_id"] != mid]
            else:
                reward -= 0.7
                # Negotiation Engine: generate a counter proposal on failure
                cp = self._generate_counter_proposal(mid)
                if cp:
                    self.counter_proposals.append(cp)
                    info["negotiation"] = f"Participant offered {cp['proposed_time']}"

        elif action.action_type == "accept_proposal" and action.proposal_id:
            cp = next((p for p in self.counter_proposals if p["proposal_id"] == action.proposal_id), None)
            if cp:
                mid = cp["meeting_id"]
                req = next((r for r in self.requests if r.id == mid), None)
                # Success is guaranteed for a CP
                self.scheduled = [m for m in self.scheduled if m["meeting_id"] != mid] # remove old if exists
                self.scheduled.append({
                    "meeting_id": mid,
                    "time": cp["proposed_time"],
                    "participants": req.participants
                })
                self._update_participant_schedules(cp["proposed_time"], req)
                reward += 0.55 # Bonus for using negotiation
                self.counter_proposals = [p for p in self.counter_proposals if p["proposal_id"] != action.proposal_id]
            else:
                reward -= 0.4 # invalid proposal id

        else:
            reward -= 0.05
            info["error"] = f"unsupported action: {action.action_type}"

        # Progress reward
        progress = len(self.scheduled) / len(self.requests) if self.requests else 0
        reward += progress * 0.2
        
        reward = max(-1.0, min(1.0, reward))
        self.total_reward += reward

        self.done = (
            self.step_count >= self.max_steps
            or (len(self.scheduled) == len(self.requests) and len(self.requests) > 0)
        )
        return self._get_observation(), reward, self.done, info

    def _validate_slot(self, proposed_time: str, req: MeetingRequest) -> Tuple[bool, float]:
        try:
            dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        except: return False, 0.0
        if dt.tzinfo is None: dt = pytz.utc.localize(dt)

        # 1. Dependency Check (Planning)
        if req.depends_on:
            dep_m = next((m for m in self.scheduled if m["meeting_id"] == req.depends_on), None)
            if not dep_m: return False, -0.5 # Mandatory planning failure
            dep_time = datetime.fromisoformat(dep_m["time"].replace("Z", "+00:00"))
            if dep_time.tzinfo is None: dep_time = pytz.utc.localize(dep_time)
            if dt <= dep_time: return False, -0.5 # Temporal violation
            
        # 2. Deadline Check
        if req.deadline_hours:
            cutoff = self.episode_start_time + timedelta(hours=req.deadline_hours)
            if dt > cutoff: return False, -0.3 # Deadline missed
            
        end_dt = dt + timedelta(minutes=req.duration_minutes)
        day_name = dt.strftime("%A")
        constraint_delta = 0.0

        for pid in req.participants:
            p = self.participants[pid]
            tz = pytz.timezone(p.timezone)
            local_dt, local_end = dt.astimezone(tz), end_dt.astimezone(tz)

            # Check Availability
            in_slot = False
            for slot in (p.availability or []):
                s_start = datetime.fromisoformat(slot["start"])
                s_end = datetime.fromisoformat(slot["end"])
                if s_start.tzinfo is None: s_start = pytz.utc.localize(s_start)
                if s_end.tzinfo is None: s_end = pytz.utc.localize(s_end)
                if local_dt >= s_start and local_end <= s_end:
                    in_slot = True
                    break
            if not in_slot: return False, 0.0

            # Check Hidden Profile Preferences
            profile = self.profiles.get(pid)
            if profile:
                if day_name in profile.avoid_days: constraint_delta -= 0.4
                
                today = dt.date().isoformat()
                todays = [m for m in self.participant_schedules.get(pid, []) if m["start"].date().isoformat() == today]
                if len(todays) >= profile.max_meetings_per_day: constraint_delta -= profile.fatigue_penalty
                
                if self.participant_schedules.get(pid):
                    last = self.participant_schedules[pid][-1]
                    if (dt - last["end"]).total_seconds() / 60 < 30:
                        constraint_delta += profile.soft_constraints.get("back_to_back", 0)

        return True, constraint_delta

    def _generate_counter_proposal(self, meeting_id: str) -> Optional[Dict]:
        """Participants check their availability and offer an alternative slot."""
        req = next((r for r in self.requests if r.id == meeting_id), None)
        if not req: return None
        
        # Try to find a slot where all participants are available
        common_slots = []
        p1 = self.participants[req.participants[0]]
        for slot in (p1.availability or []):
            s_start = datetime.fromisoformat(slot["start"])
            if s_start.tzinfo is None: s_start = pytz.utc.localize(s_start)
            
            # Check if everyone else is free at this exact start time
            all_free = True
            for pid in req.participants[1:]:
                other = self.participants[pid]
                found = False
                for other_slot in (other.availability or []):
                    os_start = datetime.fromisoformat(other_slot["start"])
                    if os_start.tzinfo is None: os_start = pytz.utc.localize(os_start)
                    if abs((s_start - os_start).total_seconds()) < 60:
                        found = True; break
                if not found: all_free = False; break
            
            if all_free:
                return {
                    "proposal_id": str(uuid.uuid4())[:8],
                    "meeting_id": meeting_id,
                    "proposed_time": s_start.isoformat(),
                    "reason": "Found a slot that works for everyone."
                }
        return None

    def _update_participant_schedules(self, proposed_time: str, req: MeetingRequest):
        dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = pytz.utc.localize(dt)
        end_dt = dt + timedelta(minutes=req.duration_minutes)
        for pid in req.participants:
            self.participant_schedules.setdefault(pid, []).append({"start": dt, "end": end_dt, "meeting_id": req.id})

    def _get_observation(self) -> "Observation":
        # Participants hide availability until profiles are read (God-tier POMDP)
        obs_participants = []
        for p in self.participants.values():
            p_copy = p.model_copy()
            if p.id not in self.profiles_read:
                p_copy.availability = None # HIDDEN
            obs_participants.append(p_copy)
            
        return Observation(
            current_time=datetime.now(pytz.UTC),
            participants=obs_participants,
            requests=self.requests,
            scheduled_meetings=self.scheduled,
            cancelled_meetings=self.cancelled_meetings,
            profiles_read=self.profiles_read,
            counter_proposals=self.counter_proposals,
            step_count=self.step_count
        )

    def state(self) -> Dict:
        return {
            "task": self.current_task, "step": self.step_count, "max_steps": self.max_steps,
            "scheduled": len(self.scheduled), "total": len(self.requests), "done": self.done
        }

    def get_grader_score(self) -> Dict:
        """God-tier multi-component grader."""
        if not self.requests: return {"score": 0.0}
        
        # 1. Completion (35%)
        comp_score = len(self.scheduled) / len(self.requests)
        
        # 2. Preference Aligment (25%)
        pref_violations = 0
        for m in self.scheduled:
            for pid in m["participants"]:
                if pid not in self.profiles_read: pref_violations += 1 # Critical: didn't even check profile
        pref_score = max(0.0, 1.0 - (pref_violations * 0.2))
        
        # 3. Dynamic Complexity (20%) - Dependency and Deadline
        complexity_violations = 0
        mid_map = {m["meeting_id"]: m["time"] for m in self.scheduled}
        for req in self.requests:
            if req.depends_on and req.id in mid_map and req.depends_on in mid_map:
                t1 = datetime.fromisoformat(mid_map[req.id].replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(mid_map[req.depends_on].replace("Z", "+00:00"))
                if t1 <= t2: complexity_violations += 1
        complexity_score = max(0.0, 1.0 - (complexity_violations * 0.5))
        
        # 4. Step Efficiency (20%)
        efficiency_score = max(0.0, 1.0 - (self.step_count / self.max_steps))
        
        final_score = (comp_score * 0.35) + (pref_score * 0.25) + (complexity_score * 0.20) + (efficiency_score * 0.20)
        
        return {
            "score": round(final_score, 3),
            "breakdown": {
                "completion": comp_score, "preferences": pref_score,
                "complexity": complexity_score, "efficiency": efficiency_score
            }
        }


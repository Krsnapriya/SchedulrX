from schedulrx.seed import set_seed
set_seed(42)

import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import numpy as np
import pytz
from models.schemas import Observation, Action, Participant, MeetingRequest, HiddenProfile


import uuid
import re
import json

# --- Utility: Standardized Time Handling ---
def to_minutes(t: str) -> int:
    """Safely convert HH:MM string to minutes since midnight."""
    try:
        h, m = map(int, t.split(":"))
        return h * 60 + m
    except (ValueError, AttributeError):
        return 0

def safe_slot(slot) -> Optional[str]:
    """Validate slot format to prevent crashes on judge-injected junk."""
    if not isinstance(slot, str) or ":" not in slot:
        return None
    return slot

def extract_time(text: str) -> Optional[str]:
    """Regex-based extraction of time constraints from natural language."""
    # Matches "10 am", "10AM", "10:00 pm", etc.
    match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', text.lower())
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    meridiem = match.group(3)

    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    return f"{hour:02d}:{minute:02d}"
# ------------------------------------------

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
        self.is_adversarial = False
        # Soft constraint tracking — visible to grader
        self.metrics = {
            "soft_constraint_violations": 0,
            "soft_constraint_successes": 0,
            "traps_triggered": [],
        }
        self.trajectory: List[Dict] = []
        self._extracted_soft_constraints: Dict[str, Dict] = {}

    # ── Adversarial task configurations ─────────────────────────────────
    ADVERSARIAL_CONFIGS = {
        "preference_conflict": {
            "description": "Hidden preference conflicts between participants",
            "traps": [
                {"participant": "p1", "type": "no_meetings_during", "hour_start": 10, "hour_end": 12, "penalty": 0.4, "visibility": "buried_in_profile"},
                {"participant": "p3", "type": "no_meetings_during", "hour_start": 14, "hour_end": 16, "penalty": 0.3, "visibility": "only_after_read"},
            ],
        },
        "deceptive_availability": {
            "description": "Availability windows that look good but violate soft constraints",
            "traps": [
                {"participant": "p2", "type": "fake_free", "day": "Wednesday", "penalty": 0.5, "visibility": "hidden"},
                {"participant": "p5", "type": "back_to_back_trap", "gap_minutes": 15, "penalty": 0.4, "visibility": "low"},
            ],
        },
        "priority_inversion": {
            "description": "Low-priority meetings block high-priority slots if scheduled first",
            "traps": [
                {"type": "slot_competition", "meetings": ["r1", "r3"], "contested_hour": 11, "penalty": 0.6},
            ],
        },
    }

    def reset(self, task_name: str = "easy", seed: int = 42) -> "Observation":
        # Determinism lock
        self._rng.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.current_task = task_name
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.scheduled = []
        self.profiles_read = {}
        self.participant_schedules = {}
        self.counter_proposals = []
        self.cancelled_meetings = []
        self.trajectory = []
        
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
            "p1": HiddenProfile(profile="Senior Lead. Prefers deep work blocks. Routine is key.", 
                                preferred_times=["morning"], avoid_days=["Saturday"],
                                max_meetings_per_day=2, fatigue_penalty=0.3,
                                soft_constraints={"back_to_back": -0.4}),
            "p2": HiddenProfile(profile="DevOps. Active late. Prefers PM focus.",
                                preferred_times=["afternoon"], avoid_days=[],
                                max_meetings_per_day=3, fatigue_penalty=0.1,
                                soft_constraints={"late_night": -0.6}),
            "p3": HiddenProfile(profile="Product Manager. Early bird. Busy Tuesdays.",
                                preferred_times=["morning"], avoid_days=["Friday"],
                                max_meetings_per_day=1, fatigue_penalty=0.5,
                                soft_constraints={"monday": -0.3}),
            "p4": HiddenProfile(profile="Designer. Flexible. Loves collaborative PM sessions.",
                                preferred_times=["afternoon"], avoid_days=[],
                                max_meetings_per_day=4, fatigue_penalty=0.2,
                                soft_constraints={}),
            "p5": HiddenProfile(profile="VP Engineering. High workload. Limit meetings.",
                                preferred_times=["evening"], avoid_days=["Sunday"],
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
        elif task_name == "hard":
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

        self.is_adversarial = False
        self.dynamic_event = None
        self.last_event = None

        if task_name == "adversarial" or task_name == "conflict" or task_name == "dynamic":
            self.is_adversarial = True
            self.max_steps = 30
            
            if task_name == "conflict":
                self._load_json_task("tasks/level2_conflict.json")
            elif task_name == "dynamic":
                self._load_json_task("tasks/level3_dynamic.json")
            elif task_name == "adversarial":
                self._load_json_task("tasks/hard_adversarial.json")
            
            # Refresh constraints
            self._extracted_soft_constraints = self._extract_soft_constraints()
            self._generate_availability(exclude_base=True)
        else:
            self._generate_availability()
            
        # Extract soft constraints from profiles (natural language parsing)
        self._extracted_soft_constraints = self._extract_soft_constraints()
        return self._get_observation()

    def _force_trap_availability(self):
        """Ensure 09:00 on Monday (April 6, 2026) is available for all."""
        from gym_env import BASE_DT
        for p in self.participants.values():
            tz = pytz.timezone(p.timezone)
            slot_start = BASE_DT.replace(hour=9, minute=0)
            slot_end = slot_start + timedelta(hours=1)
            p.availability = [{
                "start": slot_start.astimezone(tz).isoformat(),
                "end":   slot_end.astimezone(tz).isoformat(),
            }]

            # Also add an optimal slot (13:00 Monday - valid slot hour)
            opt_start = BASE_DT.replace(hour=13, minute=0)
            opt_end = opt_start + timedelta(hours=1)
            p.availability.append({
                "start": opt_start.astimezone(tz).isoformat(),
                "end":   opt_end.astimezone(tz).isoformat(),
            })

    def _extract_soft_constraints(self) -> Dict[str, Dict]:
        constraints = {}
        for pid, prof in self.profiles.items():
            text = (prof.profile + " " + " ".join(prof.history)).lower()
            c = {}
            # Robust Keyword detection
            time_limit = extract_time(text)
            if time_limit and ("not before" in text or "after" in text):
                c["not_before"] = time_limit
            elif "morning" in text and ("tough" in text or "avoid" in text or "prefer" in text):
                c["not_before"] = "10:00"
            
            if "before lunch" in text or "lunch" in text or ("afternoon" in text and ("avoid" in text or "packed" in text)):
                c["not_after"] = "12:00"

            if c:
                constraints[pid] = c
        return constraints

    def _load_json_task(self, path: str):
        """Load task definition from JSON file for Level 2/3 tasks."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # 1. Map participants to fixed p1-p5 slots and build name map
            name_to_id = {}
            new_profiles = {}
            for p_idx, p in enumerate(data.get("participants", [])):
                if p_idx >= 5: break
                pid = f"p{p_idx + 1}"
                name_to_id[p["name"]] = pid
                
                if pid in self.participants:
                    self.participants[pid].name = p["name"]
                    self.participants[pid].timezone = "UTC" # Force UTC for adversarial isolation
                    
                    from gym_env import BASE_DT, SLOT_HOURS
                    window = p.get("availability", ["09:00", "17:00"])
                    start_h = int(window[0].split(":")[0])
                    end_h = int(window[1].split(":")[0])
                    
                    avail = []
                    tz = pytz.timezone(self.participants[pid].timezone)
                    for day_off in range(5):
                        for h in SLOT_HOURS:
                            if start_h <= h < end_h:
                                slot_start = BASE_DT.replace(hour=h, minute=0, second=0) + timedelta(days=day_off)
                                slot_end = slot_start + timedelta(hours=1)
                                avail.append({
                                    "start": slot_start.astimezone(tz).isoformat(),
                                    "end":   slot_end.astimezone(tz).isoformat()
                                })
                    self.participants[pid].availability = avail

                new_profiles[pid] = HiddenProfile(
                    profile=p.get("profile", ""),
                    history=p.get("history", []),
                    preferred_times=p.get("preferred_times", []),
                    avoid_days=p.get("avoid_days", []),
                    max_meetings_per_day=p.get("max_meetings_per_day", 3),
                    fatigue_penalty=p.get("fatigue_penalty", 0.2),
                    soft_constraints=p.get("soft_constraints", {})
                )
                self.participant_schedules[pid] = []
            
            self.profiles.update(new_profiles)

            # 2. Reset requests and map participant names to IDs
            self.requests = []
            for r_data in data.get("requests", []):
                mapped_p = [name_to_id.get(p_name, p_name) for p_name in r_data.get("participants", [])]
                r_data["participants"] = mapped_p
                self.requests.append(MeetingRequest(**r_data))
            
            self.scheduled = []
            self.dynamic_event = data.get("dynamic_event")
            self._extracted_soft_constraints = self._extract_soft_constraints()
        except Exception as e:
            print(f"Error loading task {path}: {e}")

    def _inject_adversarial_constraints(self):
        """Standard L1 adversarial traps."""
        self.profiles["p1"].history.append("Alex: mornings are a bit tough for me generally, let's keep them clear before 10 AM")
        self.profiles["p3"].avoid_days.append("Tuesday")
        self.profiles["p5"].max_meetings_per_day = 1
        self.profiles["p4"].soft_constraints["back_to_back"] = -0.6

    def _check_soft_constraint_violation(self, proposed_time: str, req: MeetingRequest) -> Dict:
        """
        Adversarial penalty calculator. Uses cumulative penalty count 
        to evaluate trade-off reasoning depth.
        """
        try:
            # Extract time part HH:MM safely
            t_match = re.search(r'(\d{2}:\d{2})', proposed_time)
            if not t_match: return {"violated": False, "penalty": 0, "details": [], "count": 0}
            mins = to_minutes(t_match.group(1))
        except: return {"violated": False, "penalty": 0, "details": [], "count": 0}

        details = []
        for pid in req.participants:
            # Resolve participant ID if name was used in request
            p_id = pid if pid in self._extracted_soft_constraints else \
                   next((k for k, v in self.participants.items() if v.name == pid), None)
            
            if not p_id or p_id not in self._extracted_soft_constraints:
                continue
                
            c = self._extracted_soft_constraints[p_id]
            
            if "not_before" in c and mins < to_minutes(c["not_before"]):
                details.append(f"{pid} avoids mornings (<{c['not_before']})")
            if "not_after" in c and mins > to_minutes(c["not_after"]):
                details.append(f"{pid} avoids afternoons (>{c['not_after']})")

        return {
            "violated": len(details) > 0,
            "penalty": -0.4 * len(details),
            "details": details,
            "count": len(details)
        }

    def _generate_availability(self, exclude_base: bool = False):
        """God-tier stochastic availability: non-uniform slots."""
        for p in self.participants.values():
            tz = pytz.timezone(p.timezone)
            avail = p.availability if exclude_base and p.availability else []
            profile = self.profiles.get(p.id)
            for day in range(5):
                day_start = self.episode_start_time + timedelta(days=day)
                # Skip Monday if we already forced trap slots there
                if exclude_base and day == 0:
                    continue
                
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
        info   = {"metrics": self.metrics}
        
        # Crash-Proofing: Validate basic payload integrity
        if not action or not hasattr(action, 'action_type'):
             return self._get_observation(), -1.0, self.done, {"error": "Invalid action format"}
        
        # --- Dynamic Event Injection (Level 3) ---
        if self.dynamic_event and self.step_count == self.dynamic_event.get("trigger_step"):
            update = self.dynamic_event.get("update", {})
            p_name = update.get("participant")
            msg = update.get("message")
            
            # Find participant by name
            for pid, p in self.participants.items():
                if p.name == p_name:
                    self.profiles[pid].history.append(msg)
                    # Refresh constraints
                    self._extracted_soft_constraints = self._extract_soft_constraints()
                    self.last_event = msg
                    info["dynamic_update"] = msg
                    break

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
                # Check soft constraint violations (adversarial layer)
                soft_violation = self._check_soft_constraint_violation(m_time, req)
                if soft_violation["violated"]:
                    constraint_delta += soft_violation["penalty"]
                    info["soft_constraint_violation"] = True
                    info["soft_violation_details"] = soft_violation["details"]
                    self.metrics["soft_constraint_violations"] += 1
                    self.metrics["traps_triggered"].append({
                        "meeting": mid,
                        "time": m_time,
                        "traps": soft_violation["details"],
                    })
                else:
                    info["soft_constraint_satisfied"] = True
                    self.metrics["soft_constraint_successes"] += 1

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
        
        # Record to trajectory
        self.trajectory.append({
            "step": self.step_count,
            "action": action.model_dump() if hasattr(action, 'model_dump') else action,
            "reward": reward,
            "done": self.done,
            "info": info
        })
        
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
            step_count=self.step_count,
            last_event=self.last_event
        )

    def state(self) -> Dict:
        return {
            "task": self.current_task, "step": self.step_count, "max_steps": self.max_steps,
            "scheduled": len(self.scheduled), "total": len(self.requests), "done": self.done
        }

    def get_grader_score(self) -> Dict:
        """Structured grader with capability scores, failure modes, and trajectory summary."""
        from schedulrx.graders import programmatic_grade

        return programmatic_grade(
            requests=self.requests,
            scheduled=self.scheduled,
            profiles=self.profiles,
            profiles_read=self.profiles_read,
            participant_schedules=self.participant_schedules,
            step_count=self.step_count,
            max_steps=self.max_steps,
            cancelled_meetings=self.cancelled_meetings,
            participants=self.participants,
            episode_start_time=self.episode_start_time,
            metrics=self.metrics,
            trajectory=self.trajectory
        )


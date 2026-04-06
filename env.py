"""
SchedulrX Environment — God-Tier Edition
==========================================
What makes this genuinely hard (not just search):

1. TRUE PARTIAL OBSERVABILITY
   Participant availability is HIDDEN until the agent calls read_profile.
   read_profile is now a hard prerequisite for scheduling, not optional.

2. CASCADING MEETING DEPENDENCIES
   Some meetings have depends_on constraints — they cannot be scheduled
   before their prerequisite meeting is confirmed. Forces planning, not search.

3. PARTICIPANT COUNTER-PROPOSALS
   Scheduling at a constraint-violating time triggers a counter-proposal
   from the participant. Agent can accept it in one step (high reward)
   or ignore it and find a different slot.

4. STOCHASTIC CANCELLATIONS (hard task only)
   At a random step, a confirmed meeting gets cancelled due to an "emergency".
   Agent must detect this and reschedule. Tests replanning ability.

5. RICH MULTI-COMPONENT GRADER
   score = completion(35%) + preference_alignment(25%)
         + priority_efficiency(20%) + step_efficiency(20%)
   Greedy agents score ~0.35. Smart agents score ~0.90.
"""

import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import pytz
from models.schemas import (
    Observation, Action, Participant, MeetingRequest, HiddenProfile
)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


class SchedulrXEnv:
    def __init__(self):
        self.current_task: Optional[str] = None
        self.done: bool = False
        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.max_steps: int = 30
        self.profiles: Dict[str, HiddenProfile] = {}
        self.scheduled: List[Dict] = {}       # meeting_id → scheduled entry
        self.cancelled: List[str] = []         # cancelled meeting_ids
        self.requests: List[MeetingRequest] = []
        self.participants: Dict[str, Participant] = {}
        self.profiles_read: Dict[str, HiddenProfile] = {}
        self.participant_schedules: Dict[str, List[Dict]] = {}
        self.counter_proposals: List[Dict] = []  # active counter-proposals
        self.cancellation_step: Optional[int] = None  # step when cancellation fires
        self._rng = random.Random()
        self._cancellation_fired: bool = False

    # ------------------------------------------------------------------ reset
    def reset(self, task_name: str = "easy", seed: int = 42) -> "Observation":
        self._rng.seed(seed)
        self.current_task = task_name
        self.done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.scheduled = {}
        self.cancelled = []
        self.profiles_read = {}
        self.participant_schedules = {}
        self.counter_proposals = []
        self._cancellation_fired = False

        self.participants = {
            "p1": Participant(id="p1", name="Alice",
                              timezone="Asia/Kolkata", availability=None),  # HIDDEN
            "p2": Participant(id="p2", name="Bob",
                              timezone="UTC", availability=None),
            "p3": Participant(id="p3", name="Carol",
                              timezone="America/New_York", availability=None),
            "p4": Participant(id="p4", name="Dave",
                              timezone="Europe/London", availability=None),
            "p5": Participant(id="p5", name="Eve",
                              timezone="Asia/Tokyo", availability=None),
        }

        # TRUE hidden profiles — not exposed until read_profile is called
        self.profiles = {
            "p1": HiddenProfile(
                preferred_times=["morning"], avoid_days=["Saturday"],
                max_meetings_per_day=2, fatigue_penalty=0.35,
                soft_constraints={"back_to_back": -0.4}
            ),
            "p2": HiddenProfile(
                preferred_times=["afternoon"], avoid_days=[],
                max_meetings_per_day=3, fatigue_penalty=0.1,
                soft_constraints={"late_night": -0.6}
            ),
            "p3": HiddenProfile(
                preferred_times=["morning"], avoid_days=["Friday"],
                max_meetings_per_day=1, fatigue_penalty=0.5,
                soft_constraints={"monday": -0.3}
            ),
            "p4": HiddenProfile(
                preferred_times=["afternoon"], avoid_days=[],
                max_meetings_per_day=4, fatigue_penalty=0.15,
                soft_constraints={}
            ),
            "p5": HiddenProfile(
                preferred_times=["evening"], avoid_days=["Sunday"],
                max_meetings_per_day=2, fatigue_penalty=0.4,
                soft_constraints={"back_to_back": -0.5}
            ),
        }

        # Generate hidden availability (not exposed until profile read)
        self._hidden_availability: Dict[str, List[Dict]] = {}
        self._generate_availability()

        for pid in self.participants:
            self.participant_schedules[pid] = []

        if task_name == "easy":
            self.requests = [
                MeetingRequest(
                    id="r1", title="Team sync",
                    duration_minutes=30, priority=8,
                    participants=["p1", "p2"],
                    depends_on=None,
                    deadline_hours=48,
                )
            ]
            self.cancellation_step = None  # no cancellations on easy

        elif task_name == "medium":
            self.requests = [
                MeetingRequest(
                    id="r1", title="Project kickoff",
                    duration_minutes=45, priority=9,
                    participants=["p1", "p2", "p3"],
                    depends_on=None,
                    deadline_hours=72,
                ),
                MeetingRequest(
                    id="r2", title="Design review",
                    duration_minutes=60, priority=7,
                    participants=["p2", "p4"],
                    depends_on="r1",           # DEPENDENCY: r1 must be scheduled first
                    deadline_hours=96,
                ),
                MeetingRequest(
                    id="r3", title="Budget check",
                    duration_minutes=30, priority=6,
                    participants=["p1", "p5"],
                    depends_on=None,
                    deadline_hours=120,
                ),
            ]
            self.cancellation_step = None

        else:  # hard
            self.requests = [
                MeetingRequest(
                    id="r1", title="Strategy offsite",
                    duration_minutes=60, priority=10,
                    participants=["p1", "p2", "p3", "p4"],
                    depends_on=None,
                    deadline_hours=48,
                ),
                MeetingRequest(
                    id="r2", title="Investor call",
                    duration_minutes=60, priority=9,
                    participants=["p1", "p3", "p5"],
                    depends_on="r1",           # DEPENDENCY
                    deadline_hours=72,
                ),
                MeetingRequest(
                    id="r3", title="Team retro",
                    duration_minutes=45, priority=5,
                    participants=["p2", "p4"],
                    depends_on=None,
                    deadline_hours=120,
                ),
            ]
            # Stochastic cancellation fires between step 8 and 14
            self.cancellation_step = self._rng.randint(8, 14)

        return self._get_observation()

    # -------------------------------------------------------- availability gen
    def _generate_availability(self):
        """
        Generate realistic, SPARSE availability per participant.
        Each participant gets 3–5 slots per day (not all the same).
        Stored hidden — only revealed when read_profile is called.
        """
        now = datetime.now(pytz.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        all_hours = [9, 10, 11, 13, 14, 15, 16]

        for pid, p in self.participants.items():
            tz = pytz.timezone(p.timezone)
            profile = self.profiles[pid]
            avail = []

            for day in range(5):
                day_start = now + timedelta(days=day)
                day_name = day_start.strftime("%A")

                # Skip avoid_days entirely
                if day_name in profile.avoid_days:
                    continue

                # Each participant gets a different subset of hours
                # seeded by pid so it's reproducible but varied
                pid_seed = sum(ord(c) for c in pid)
                day_hours = [
                    h for h in all_hours
                    if (pid_seed + day + h) % 3 != 0   # removes ~33% of slots
                ]

                for hour in day_hours:
                    slot_start = day_start.replace(hour=hour, minute=0)
                    slot_end = day_start.replace(hour=hour + 1, minute=30)
                    avail.append({
                        "start": slot_start.astimezone(tz).isoformat(),
                        "end":   slot_end.astimezone(tz).isoformat(),
                    })

            self._hidden_availability[pid] = avail

    # ------------------------------------------------------------------- step
    def step(self, action: Action) -> Tuple["Observation", float, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        info = {}

        # ── Stochastic cancellation check (hard task) ──────────────────────
        if (self.cancellation_step is not None
                and self.step_count == self.cancellation_step
                and not self._cancellation_fired):
            self._fire_cancellation(info)

        # ── Action dispatch ────────────────────────────────────────────────
        if action.action_type == "read_profile":
            reward = self._handle_read_profile(action, info)

        elif action.action_type == "schedule_meeting":
            reward = self._handle_schedule(action, info)

        elif action.action_type == "accept_proposal":
            reward = self._handle_accept_proposal(action, info)

        elif action.action_type == "reschedule_meeting":
            reward = self._handle_reschedule(action, info)

        else:
            reward = -0.05
            info["error"] = f"unknown action_type: {action.action_type}"

        # ── Progress bonus (dense shaped) ─────────────────────────────────
        active_requests = [r for r in self.requests if r.id not in self.cancelled]
        n_done = sum(1 for r in active_requests if r.id in self.scheduled)
        progress = n_done / max(len(active_requests), 1)
        reward += progress * 0.2
        reward = max(-1.0, min(1.0, reward))
        self.total_reward += reward

        self.done = (
            self.step_count >= self.max_steps
            or n_done == len(active_requests)
        )
        return self._get_observation(), reward, self.done, info

    # ─────────────────────────────────── action handlers
    def _handle_read_profile(self, action: Action, info: Dict) -> float:
        pid = action.participant_id
        if not pid or pid not in self.profiles:
            return -0.15

        if pid in self.profiles_read:
            return -0.1   # already read

        # REVEAL availability + profile
        self.profiles_read[pid] = self.profiles[pid]
        self.participants[pid].availability = self._hidden_availability[pid]
        info["discovered"] = pid
        info["slots_revealed"] = len(self._hidden_availability[pid])
        return 0.25

    def _handle_schedule(self, action: Action, info: Dict) -> float:
        mid = action.meeting_id
        proposed_time = action.proposed_time

        if not mid or not proposed_time:
            return -0.1

        # Already scheduled
        if mid in self.scheduled:
            return -0.8

        # Cancelled
        if mid in self.cancelled:
            return -0.3

        req = next((r for r in self.requests if r.id == mid), None)
        if not req:
            return -0.6

        # ── Dependency check ───────────────────────────────────────────────
        if req.depends_on and req.depends_on not in self.scheduled:
            info["error"] = f"dependency {req.depends_on} not yet scheduled"
            return -0.5   # agent tried to skip dependency

        # ── Profile read check ─────────────────────────────────────────────
        # Agent MUST have read profiles of all participants before scheduling
        unread = [p for p in req.participants if p not in self.profiles_read]
        if unread:
            info["error"] = f"profiles not read for: {unread}"
            info["hint"] = "call read_profile for each participant first"
            return -0.4   # hard block — must read profiles first

        # ── Slot validation ────────────────────────────────────────────────
        valid, constraint_delta, violation_detail = self._validate_slot(
            proposed_time, req
        )

        if not valid:
            # Check if we should generate a counter-proposal
            proposal = self._generate_counter_proposal(req, proposed_time)
            if proposal:
                self.counter_proposals.append(proposal)
                info["counter_proposal"] = proposal
                info["error"] = "slot invalid — counter-proposal offered"
                return -0.25   # softer penalty when counter-proposal available

            info["error"] = f"invalid slot: {violation_detail}"
            return -0.7

        # ── Success ────────────────────────────────────────────────────────
        self.scheduled[mid] = {
            "meeting_id": mid,
            "time": proposed_time,
            "participants": req.participants,
            "priority": req.priority,
        }
        self._update_participant_schedules(proposed_time, req)

        # Remove any counter-proposals for this meeting
        self.counter_proposals = [
            cp for cp in self.counter_proposals
            if cp["meeting_id"] != mid
        ]

        base_reward = 0.45 + constraint_delta
        info["scheduled"] = mid
        info["constraint_delta"] = round(constraint_delta, 3)
        return base_reward

    def _handle_accept_proposal(self, action: Action, info: Dict) -> float:
        """Accept a counter-proposal generated by the environment."""
        proposal_id = action.proposal_id
        if not proposal_id:
            return -0.1

        proposal = next(
            (cp for cp in self.counter_proposals if cp["proposal_id"] == proposal_id),
            None
        )
        if not proposal:
            return -0.2   # proposal expired or doesn't exist

        mid = proposal["meeting_id"]
        proposed_time = proposal["proposed_time"]
        req = next((r for r in self.requests if r.id == mid), None)

        if not req or mid in self.scheduled:
            return -0.2

        # Counter-proposals are already validated — accept is near-guaranteed
        valid, constraint_delta, _ = self._validate_slot(proposed_time, req)
        if valid:
            self.scheduled[mid] = {
                "meeting_id": mid,
                "time": proposed_time,
                "participants": req.participants,
                "priority": req.priority,
            }
            self._update_participant_schedules(proposed_time, req)
            self.counter_proposals = [
                cp for cp in self.counter_proposals
                if cp["meeting_id"] != mid
            ]
            info["accepted_proposal"] = proposal_id
            return 0.55   # bonus for negotiating efficiently

        return -0.1

    def _handle_reschedule(self, action: Action, info: Dict) -> float:
        """Reschedule a cancelled meeting."""
        mid = action.meeting_id
        proposed_time = action.proposed_time

        if mid not in self.cancelled:
            return -0.2   # meeting wasn't cancelled

        if mid in self.scheduled:
            return -0.3   # already rescheduled

        req = next((r for r in self.requests if r.id == mid), None)
        if not req:
            return -0.4

        unread = [p for p in req.participants if p not in self.profiles_read]
        if unread:
            return -0.3

        valid, constraint_delta, violation_detail = self._validate_slot(
            proposed_time, req
        )
        if valid:
            self.scheduled[mid] = {
                "meeting_id": mid,
                "time": proposed_time,
                "participants": req.participants,
                "priority": req.priority,
            }
            self._update_participant_schedules(proposed_time, req)
            self.cancelled.remove(mid)  # back to active
            info["rescheduled"] = mid
            return 0.50   # high reward for recovering from cancellation
        else:
            info["error"] = f"reschedule failed: {violation_detail}"
            return -0.5

    # ─────────────────────────────────── slot logic
    def _validate_slot(
        self, proposed_time: str, req: MeetingRequest
    ) -> Tuple[bool, float, str]:
        try:
            dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return False, 0.0, "unparseable datetime"

        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)

        duration = timedelta(minutes=req.duration_minutes)
        end_dt = dt + duration
        day_name = dt.strftime("%A")
        constraint_delta = 0.0

        for pid in req.participants:
            p = self.participants.get(pid)
            if not p or p.availability is None:
                return False, 0.0, f"availability hidden for {pid} — read profile first"

            tz = pytz.timezone(p.timezone)
            local_dt = dt.astimezone(tz)
            local_end = end_dt.astimezone(tz)

            in_slot = False
            for slot in p.availability:
                try:
                    s_start = datetime.fromisoformat(slot["start"])
                    s_end = datetime.fromisoformat(slot["end"])
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
                return False, 0.0, f"{pid} not available at {proposed_time}"

            profile = self.profiles.get(pid)
            if profile:
                # Hard avoid_days → soft penalty (not block — agent discovered it)
                if day_name in profile.avoid_days:
                    constraint_delta -= 0.35

                # Per-day fatigue check
                today_str = dt.date().isoformat()
                todays = [
                    m for m in self.participant_schedules.get(pid, [])
                    if m["start"].date().isoformat() == today_str
                ]
                if len(todays) >= profile.max_meetings_per_day:
                    constraint_delta -= profile.fatigue_penalty

                # Back-to-back check
                if self.participant_schedules.get(pid):
                    last = self.participant_schedules[pid][-1]
                    gap = (dt - last["end"]).total_seconds() / 60
                    if gap < 30:
                        constraint_delta += profile.soft_constraints.get(
                            "back_to_back", 0
                        )

        return True, constraint_delta, ""

    def _generate_counter_proposal(
        self, req: MeetingRequest, failed_time: str
    ) -> Optional[Dict]:
        """
        After a scheduling failure, try to find a valid slot and offer
        it as a counter-proposal. Returns None if no alternative found.
        """
        import uuid as _uuid
        try:
            failed_dt = datetime.fromisoformat(failed_time.replace("Z", "+00:00"))
        except Exception:
            return None

        # Try a few nearby slots
        for delta_hours in [2, 4, 24, 26, 48]:
            candidate = failed_dt + timedelta(hours=delta_hours)
            candidate_iso = candidate.isoformat()
            valid, _, _ = self._validate_slot(candidate_iso, req)
            if valid:
                return {
                    "proposal_id": str(_uuid.uuid4())[:8],
                    "meeting_id": req.id,
                    "proposed_time": candidate_iso,
                    "reason": f"alternative to {failed_time}",
                }
        return None

    def _fire_cancellation(self, info: Dict):
        """Cancel a confirmed meeting — stochastic hard-task disruption."""
        self._cancellation_fired = True
        if not self.scheduled:
            return

        # Cancel the lowest-priority confirmed meeting
        scheduled_list = list(self.scheduled.values())
        scheduled_list.sort(key=lambda m: m.get("priority", 0))
        victim = scheduled_list[0]
        mid = victim["meeting_id"]

        # Remove from scheduled, add to cancelled, clear participant schedules
        del self.scheduled[mid]
        self.cancelled.append(mid)
        for pid in victim["participants"]:
            self.participant_schedules[pid] = [
                m for m in self.participant_schedules[pid]
                if m["meeting_id"] != mid
            ]
        info["cancellation"] = {
            "meeting_id": mid,
            "reason": "participant emergency — please reschedule",
        }

    def _update_participant_schedules(self, proposed_time: str, req: MeetingRequest):
        dt = datetime.fromisoformat(proposed_time.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        end_dt = dt + timedelta(minutes=req.duration_minutes)
        for pid in req.participants:
            self.participant_schedules.setdefault(pid, []).append({
                "start": dt,
                "end": end_dt,
                "meeting_id": req.id,
            })

    # ─────────────────────────────────── observation
    def _get_observation(self) -> "Observation":
        return Observation(
            current_time=datetime.now(pytz.UTC),
            participants=list(self.participants.values()),
            requests=self.requests,
            scheduled_meetings=list(self.scheduled.values()),
            cancelled_meetings=self.cancelled,
            profiles_read=self.profiles_read,
            counter_proposals=self.counter_proposals,
            step_count=self.step_count,
        )

    def state(self) -> Dict:
        active = [r for r in self.requests if r.id not in self.cancelled]
        return {
            "task": self.current_task,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "scheduled": len(self.scheduled),
            "cancelled": len(self.cancelled),
            "active_requests": len(active),
            "profiles_read": list(self.profiles_read.keys()),
            "pending_proposals": len(self.counter_proposals),
            "done": self.done,
        }

    # ─────────────────────────────────── grader
    def get_grader_score(self) -> Dict:
        """
        God-tier multi-component grader.

        completion        (35%) — fraction of active meetings scheduled
        preference_align  (25%) — how well scheduled times match hidden prefs
        priority_order    (20%) — were high-priority meetings scheduled first?
        step_efficiency   (20%) — bonus for solving in fewer steps
        """
        active = [r for r in self.requests if r.id not in self.cancelled]
        n_total = len(active)
        if n_total == 0:
            return {"score": 1.0, "breakdown": {}}

        # 1. Completion
        n_done = sum(1 for r in active if r.id in self.scheduled)
        completion = n_done / n_total

        # 2. Preference alignment
        alignment_scores = []
        for r in active:
            if r.id not in self.scheduled:
                alignment_scores.append(0.0)
                continue
            entry = self.scheduled[r.id]
            meeting_dt = datetime.fromisoformat(
                entry["time"].replace("Z", "+00:00")
            )
            if meeting_dt.tzinfo is None:
                meeting_dt = pytz.utc.localize(meeting_dt)
            score_parts = []
            for pid in r.participants:
                if pid not in self.profiles_read:
                    score_parts.append(0.5)   # unknown — neutral
                    continue
                profile = self.profiles[pid]
                tz = pytz.timezone(self.participants[pid].timezone)
                local_hour = meeting_dt.astimezone(tz).hour
                pref = profile.preferred_times[0] if profile.preferred_times else None
                if pref == "morning" and 8 <= local_hour < 12:
                    score_parts.append(1.0)
                elif pref == "afternoon" and 12 <= local_hour < 17:
                    score_parts.append(1.0)
                elif pref == "evening" and 17 <= local_hour < 21:
                    score_parts.append(1.0)
                else:
                    score_parts.append(0.3)
            alignment_scores.append(
                sum(score_parts) / len(score_parts) if score_parts else 0.5
            )
        preference_align = sum(alignment_scores) / n_total

        # 3. Priority order — were higher-priority meetings scheduled first?
        if len(self.scheduled) >= 2:
            scheduled_order = sorted(
                [self.scheduled[r.id] for r in active if r.id in self.scheduled],
                key=lambda m: m.get("time", "")
            )
            # Check if priority descends along schedule order
            priorities = [m.get("priority", 5) for m in scheduled_order]
            inversions = sum(
                1 for i in range(len(priorities) - 1)
                if priorities[i] < priorities[i + 1]
            )
            max_inversions = len(priorities) * (len(priorities) - 1) / 2
            priority_order = 1.0 - (inversions / max_inversions) if max_inversions > 0 else 1.0
        else:
            priority_order = 1.0 if n_done >= 1 else 0.0

        # 4. Step efficiency
        if n_done == n_total:
            efficiency = max(0.0, 1.0 - (self.step_count / self.max_steps))
        else:
            efficiency = 0.0

        score = (
            completion      * 0.35
            + preference_align * 0.25
            + priority_order   * 0.20
            + efficiency       * 0.20
        )

        return {
            "score": round(min(score, 1.0), 3),
            "breakdown": {
                "completion":        round(completion, 3),
                "preference_align":  round(preference_align, 3),
                "priority_order":    round(priority_order, 3),
                "step_efficiency":   round(efficiency, 3),
            },
            "completed": n_done,
            "total":     n_total,
            "cancelled": len(self.cancelled),
        }

"""
SchedulrX Gymnasium Wrapper
============================
Formal POMDP formulation of the scheduling benchmark.

State encoding: [availability_matrix | scheduled_slots | profiles_read_mask |
                 constraint_features | request_completion | step_progress]

Action space: Discrete(65)
  - Actions 0-4:  read_profile(p1..p5)
  - Actions 5-64: schedule_meeting(meeting_idx * 20 + slot_idx)
    → 3 meetings × 20 time slots (5 days × 4 blocks/day)

Observation space: Box(low=0, high=1, shape=(OBS_DIM,))
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Optional, Tuple

from env import SchedulrXEnv
from models.schemas import Action

# --- Constants ---
NUM_PARTICIPANTS = 5
NUM_DAYS = 5
SLOTS_PER_DAY = 4       # hours 9, 11, 13, 15
NUM_SLOTS = NUM_DAYS * SLOTS_PER_DAY  # 20
MAX_MEETINGS = 3
NUM_READ_ACTIONS = NUM_PARTICIPANTS            # 5
NUM_SCHEDULE_ACTIONS = MAX_MEETINGS * NUM_SLOTS  # 60
NUM_RESCHEDULE_ACTIONS = MAX_MEETINGS * NUM_SLOTS # 60
NUM_ACCEPT_ACTIONS = MAX_MEETINGS              # 3
ACTION_DIM = NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS + NUM_RESCHEDULE_ACTIONS + NUM_ACCEPT_ACTIONS # 128

# Observation dimensions
AVAIL_DIM = NUM_PARTICIPANTS * NUM_SLOTS       # 100
SCHED_DIM = NUM_PARTICIPANTS * NUM_SLOTS       # 100
PROFILE_MASK_DIM = NUM_PARTICIPANTS            # 5
CONSTRAINT_DIM = NUM_PARTICIPANTS * 4          # 20  (pref_morn, pref_aft, pref_eve, fatigue)
REQUEST_DONE_DIM = MAX_MEETINGS               # 3
STEP_DIM = 1
OBS_DIM = AVAIL_DIM + SCHED_DIM + PROFILE_MASK_DIM + CONSTRAINT_DIM + REQUEST_DONE_DIM + STEP_DIM  # 229

# Base datetime for slot indexing
BASE_DT = datetime(2026, 4, 6, 9, 0, tzinfo=pytz.UTC)
SLOT_HOURS = [9, 11, 13, 15]

PARTICIPANT_IDS = ["p1", "p2", "p3", "p4", "p5"]
PREF_MAP = {"morning": 0, "afternoon": 1, "evening": 2}


def slot_index_to_iso(slot_idx: int) -> str:
    """Convert flat slot index (0-19) to ISO datetime string."""
    day = int(slot_idx // SLOTS_PER_DAY)
    hour_idx = int(slot_idx % SLOTS_PER_DAY)
    hour = SLOT_HOURS[hour_idx]
    dt = BASE_DT + timedelta(days=day)
    dt = dt.replace(hour=hour, minute=0, second=0)
    return dt.isoformat()


def iso_to_slot_index(iso_str: str) -> int:
    """Convert ISO datetime string to flat slot index (0-19). Returns -1 if no match."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(pytz.UTC)
        day = (dt - BASE_DT).days
        if day < 0 or day >= NUM_DAYS:
            return -1
        hour = dt.hour
        if hour not in SLOT_HOURS:
            return -1
        hour_idx = SLOT_HOURS.index(hour)
        return day * SLOTS_PER_DAY + hour_idx
    except Exception:
        return -1


class SchedulrXGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for SchedulrX.
    
    Formalized as a POMDP:
    - Hidden state: participant preferences, fatigue thresholds
    - Observation: availability, scheduled meetings, discovered constraints
    - Actions: read_profile or schedule_meeting
    - Reward: dense shaped, bounded to [-1, 1]
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, task_name: str = "easy", render_mode: Optional[str] = None):
        super().__init__()
        self.task_name = task_name
        self.render_mode = render_mode
        self.core_env = SchedulrXEnv()
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)
        
        # Track request IDs for the current task
        self._request_ids = []
        self._last_info = {}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        task = self.task_name
        if options and "task_name" in options:
            task = options["task_name"]
            
        obs_pydantic = self.core_env.reset(task)
        obs_dict = obs_pydantic.model_dump()
        
        self._request_ids = [r["id"] for r in obs_dict.get("requests", [])]
        self._last_info = {}
        
        encoded = self._encode_obs(obs_dict)
        return encoded, {"raw_observation": obs_dict}
    
    def step(self, action: int):
        decoded_action = self._decode_action(action)
        
        if decoded_action is None:
            # Invalid action → penalty
            obs_dict = self.core_env._get_observation().model_dump()
            return self._encode_obs(obs_dict), -0.5, False, False, {"invalid_action": True}
        
        action_pydantic = Action(**decoded_action)
        obs_pydantic, reward, done, info = self.core_env.step(action_pydantic)
        obs_dict = obs_pydantic.model_dump()
        
        info["decoded_action"] = decoded_action
        info["raw_observation"] = obs_dict
        self._last_info = info
        
        encoded = self._encode_obs(obs_dict)
        return encoded, float(reward), bool(done), False, info
    
    def _get_obs(self) -> np.ndarray:
        """Helper to get current encoded observation."""
        obs_dict = self.core_env._get_observation().model_dump()
        return self._encode_obs(obs_dict)
    
    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of valid actions.
        Critical for preventing invalid moves during training.
        """
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        
        obs_dict = self.core_env._get_observation().model_dump()
        profiles_read = set(obs_dict.get("profiles_read", {}).keys())
        scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}
        
        # read_profile actions: valid if participant exists and not yet read
        for i, pid in enumerate(PARTICIPANT_IDS):
            if pid not in profiles_read:
                mask[i] = True
                
        # schedule_meeting actions: valid if meeting not already scheduled
        for m_idx, req_id in enumerate(self._request_ids):
            if req_id in scheduled_ids:
                continue
            for slot_idx in range(NUM_SLOTS):
                action_idx = NUM_READ_ACTIONS + m_idx * NUM_SLOTS + slot_idx
                mask[action_idx] = True

        # reschedule_meeting actions: valid if meeting is cancelled
        cancelled_ids = obs_dict.get("cancelled_meetings", [])
        for m_idx, req_id in enumerate(self._request_ids):
            if req_id in cancelled_ids:
                for slot_idx in range(NUM_SLOTS):
                    action_idx = NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS + m_idx * NUM_SLOTS + slot_idx
                    mask[action_idx] = True
                    
        # accept_proposal actions: valid if there is an active proposal
        for p in obs_dict.get("counter_proposals", []):
            m_id = p.get("meeting_id")
            if m_id in self._request_ids:
                m_idx = self._request_ids.index(m_id)
                action_idx = NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS + NUM_RESCHEDULE_ACTIONS + m_idx
                mask[action_idx] = True
                
        # Pad remaining meeting slots if fewer than MAX_MEETINGS requests
        # (they stay False — already handled by range)
        
        # Ensure at least one action is valid (fallback to read_profile)
        if not mask.any():
            mask[0] = True
            
        return mask
    
    def get_grader_score(self) -> Dict:
        return self.core_env.get_grader_score()
    
    def _encode_obs(self, obs_dict: dict) -> np.ndarray:
        """Encode raw observation dict into flat float32 vector."""
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        offset = 0
        
        # 1. Availability matrix [NUM_PARTICIPANTS × NUM_SLOTS]
        participants = obs_dict.get("participants", [])
        for p_idx, p in enumerate(participants):
            if p_idx >= NUM_PARTICIPANTS:
                break
            avail = p.get("availability")
            if avail is not None:
                for slot in avail:
                    s_idx = iso_to_slot_index(slot.get("start", ""))
                    if 0 <= s_idx < NUM_SLOTS:
                        vec[offset + p_idx * NUM_SLOTS + s_idx] = 1.0
        offset += AVAIL_DIM
        
        # 2. Scheduled slots matrix [NUM_PARTICIPANTS × NUM_SLOTS]
        for m in obs_dict.get("scheduled_meetings", []):
            s_idx = iso_to_slot_index(m.get("time", ""))
            if 0 <= s_idx < NUM_SLOTS:
                for pid in m.get("participants", []):
                    p_idx = PARTICIPANT_IDS.index(pid) if pid in PARTICIPANT_IDS else -1
                    if 0 <= p_idx < NUM_PARTICIPANTS:
                        vec[offset + p_idx * NUM_SLOTS + s_idx] = 1.0
        offset += SCHED_DIM
        
        # 3. Profile read mask [NUM_PARTICIPANTS]
        profiles_read = obs_dict.get("profiles_read", {})
        for i, pid in enumerate(PARTICIPANT_IDS):
            if pid in profiles_read:
                vec[offset + i] = 1.0
        offset += PROFILE_MASK_DIM
        
        # 4. Constraint features [NUM_PARTICIPANTS × 4]
        for i, pid in enumerate(PARTICIPANT_IDS):
            if pid in profiles_read:
                prof = profiles_read[pid]
                prefs = prof.get("preferred_times", []) if isinstance(prof, dict) else prof.preferred_times if hasattr(prof, 'preferred_times') else []
                fatigue = prof.get("fatigue_penalty", 0.0) if isinstance(prof, dict) else prof.fatigue_penalty if hasattr(prof, 'fatigue_penalty') else 0.0
                
                if isinstance(prefs, list):
                    for pref in prefs:
                        if pref in PREF_MAP:
                            vec[offset + i * 4 + PREF_MAP[pref]] = 1.0
                vec[offset + i * 4 + 3] = float(fatigue)
        offset += CONSTRAINT_DIM
        
        # 5. Request completion flags [MAX_MEETINGS]
        scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}
        for i, req_id in enumerate(self._request_ids):
            if i >= MAX_MEETINGS:
                break
            if req_id in scheduled_ids:
                vec[offset + i] = 1.0
        offset += REQUEST_DONE_DIM
        
        # 6. Step progress [1]
        max_steps = self.core_env.max_steps
        vec[offset] = obs_dict.get("step_count", 0) / max_steps
        
        return vec
    
    def _decode_action(self, action: int) -> Optional[dict]:
        """Decode flat action index into JSON action dict."""
        if action < 0 or action >= ACTION_DIM:
            return None
            
        # read_profile actions: 0..4
        if action < NUM_READ_ACTIONS:
            pid = PARTICIPANT_IDS[action]
            return {"action_type": "read_profile", "participant_id": pid}
        
        # schedule_meeting actions: 5..64
        if action < NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS:
            schedule_idx = action - NUM_READ_ACTIONS
            meeting_idx = schedule_idx // NUM_SLOTS
            slot_idx = schedule_idx % NUM_SLOTS
            
            if meeting_idx >= len(self._request_ids):
                return None
                
            meeting_id = self._request_ids[meeting_idx]
            proposed_time = slot_index_to_iso(slot_idx)
            
            return {
                "action_type": "schedule_meeting",
                "meeting_id": meeting_id,
                "proposed_time": proposed_time,
            }

        # reschedule_meeting actions: 65..124
        if action < NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS + NUM_RESCHEDULE_ACTIONS:
            reschedule_idx = action - (NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS)
            meeting_idx = reschedule_idx // NUM_SLOTS
            slot_idx = reschedule_idx % NUM_SLOTS
            
            if meeting_idx >= len(self._request_ids):
                return None
                
            meeting_id = self._request_ids[meeting_idx]
            proposed_time = slot_index_to_iso(slot_idx)
            
            return {
                "action_type": "reschedule_meeting",
                "meeting_id": meeting_id,
                "proposed_time": proposed_time,
            }

        # accept_proposal actions: 125..127
        accept_idx = action - (NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS + NUM_RESCHEDULE_ACTIONS)
        if accept_idx >= len(self._request_ids):
            return None
            
        meeting_id = self._request_ids[accept_idx]
        
        # Find proposal id from state
        for p in self.core_env.counter_proposals:
            if p.get("meeting_id") == meeting_id:
                return {
                    "action_type": "accept_proposal",
                    "proposal_id": p.get("proposal_id")
                }
        return None


class CurriculumSchedulrXEnv(SchedulrXGymEnv):
    """
    Curriculum learning wrapper.
    Automatically advances difficulty when success_rate exceeds threshold.
    """
    
    LEVELS = ["easy", "medium", "hard"]
    
    def __init__(self, advance_threshold: float = 0.7, render_mode=None):
        super().__init__(task_name="easy", render_mode=render_mode)
        self.advance_threshold = advance_threshold
        self.current_level = 0
        self.episode_scores = []
        self.episodes_per_eval = 20
        
    def reset(self, seed=None, options=None):
        # Auto-advance curriculum
        if len(self.episode_scores) >= self.episodes_per_eval:
            recent = self.episode_scores[-self.episodes_per_eval:]
            avg = sum(recent) / len(recent)
            if avg >= self.advance_threshold and self.current_level < len(self.LEVELS) - 1:
                self.current_level += 1
                self.episode_scores = []
                
        self.task_name = self.LEVELS[self.current_level]
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if done:
            score = self.get_grader_score().get("score", 0.0)
            self.episode_scores.append(score)
        return obs, reward, done, truncated, info

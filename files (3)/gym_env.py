"""
SchedulrX Gymnasium Wrapper
============================
Formal POMDP encoding of the scheduling task for direct RL training.

Observation vector (229-dim float32):
  [availability_matrix | scheduled_slots | profiles_read_mask |
   constraint_features | request_completion | step_progress]

Action space: Discrete(65)
  0–4:   read_profile(p1..p5)
  5–64:  schedule_meeting(meeting_idx * 20 + slot_idx)
         3 meetings × 20 slots (7 days × 4 blocks per day, first 5 days indexed)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pytz

from env import SchedulrXEnv        # root-level import (not server.env)
from models.schemas import Action

NUM_PARTICIPANTS      = 5
NUM_DAYS              = 5
SLOTS_PER_DAY         = 4           # four 2-hour blocks per day
NUM_SLOTS             = NUM_DAYS * SLOTS_PER_DAY   # 20
MAX_MEETINGS          = 3
NUM_READ_ACTIONS      = NUM_PARTICIPANTS            # 5
NUM_SCHEDULE_ACTIONS  = MAX_MEETINGS * NUM_SLOTS    # 60
ACTION_DIM            = NUM_READ_ACTIONS + NUM_SCHEDULE_ACTIONS  # 65

AVAIL_DIM         = NUM_PARTICIPANTS * NUM_SLOTS    # 100
SCHED_DIM         = NUM_PARTICIPANTS * NUM_SLOTS    # 100
PROFILE_MASK_DIM  = NUM_PARTICIPANTS                # 5
CONSTRAINT_DIM    = NUM_PARTICIPANTS * 4            # 20
REQUEST_DONE_DIM  = MAX_MEETINGS                    # 3
STEP_DIM          = 1
OBS_DIM           = (
    AVAIL_DIM + SCHED_DIM + PROFILE_MASK_DIM
    + CONSTRAINT_DIM + REQUEST_DONE_DIM + STEP_DIM  # 229
)

PARTICIPANT_IDS = ["p1", "p2", "p3", "p4", "p5"]
PREF_MAP        = {"morning": 0, "afternoon": 1, "evening": 2}
BLOCK_HOURS     = [8, 10, 14, 16]   # must match _generate_availability in env.py


def _base_dt() -> datetime:
    """Dynamic base: always tomorrow at 08:00 UTC."""
    return (
        datetime.now(pytz.UTC)
        .replace(hour=8, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    )


def slot_index_to_iso(slot_idx: int) -> str:
    base  = _base_dt()
    day   = slot_idx // SLOTS_PER_DAY
    hour  = BLOCK_HOURS[slot_idx % SLOTS_PER_DAY]
    dt    = (base + timedelta(days=day)).replace(hour=hour, minute=0, second=0)
    return dt.isoformat()


def iso_to_slot_index(iso_str: str) -> int:
    try:
        dt   = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(pytz.UTC)
        base = _base_dt()
        day  = (dt.date() - base.date()).days
        if day < 0 or day >= NUM_DAYS:
            return -1
        if dt.hour not in BLOCK_HOURS:
            return -1
        return day * SLOTS_PER_DAY + BLOCK_HOURS.index(dt.hour)
    except Exception:
        return -1


class SchedulrXGymEnv(gym.Env):
    """
    Gymnasium wrapper for SchedulrX.

    POMDP formulation:
      Hidden state  — participant avoid_days, fatigue thresholds, preferences
      Observation   — availability blocks, scheduled meetings, discovered constraints
      Actions       — read_profile (information gathering) or schedule_meeting
      Reward        — dense shaped, bounded [-1, 1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_name: str = "easy", render_mode: Optional[str] = None):
        super().__init__()
        self.task_name   = task_name
        self.render_mode = render_mode
        self.core_env    = SchedulrXEnv()
        self._request_ids: list = []
        self._last_info: dict   = {}

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        task     = (options or {}).get("task_name", self.task_name)
        obs_p    = self.core_env.reset(task)
        obs_dict = obs_p.model_dump()
        self._request_ids = [r["id"] for r in obs_dict.get("requests", [])]
        self._last_info   = {}
        return self._encode_obs(obs_dict), {"raw_observation": obs_dict}

    def step(self, action: int):
        decoded = self._decode_action(action)
        if decoded is None:
            obs_dict = self.core_env._get_observation().model_dump()
            return self._encode_obs(obs_dict), -0.5, False, False, {"invalid_action": True}

        pydantic_action = Action(**decoded)
        obs_p, reward, done, info = self.core_env.step(pydantic_action)
        obs_dict = obs_p.model_dump()
        info["decoded_action"]  = decoded
        info["raw_observation"] = obs_dict
        self._last_info = info
        return self._encode_obs(obs_dict), float(reward), bool(done), False, info

    def get_action_mask(self) -> np.ndarray:
        mask     = np.zeros(ACTION_DIM, dtype=np.bool_)
        obs_dict = self.core_env._get_observation().model_dump()
        profiles_read  = set(obs_dict.get("profiles_read", {}).keys())
        scheduled_ids  = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}

        for i, pid in enumerate(PARTICIPANT_IDS):
            if pid not in profiles_read:
                mask[i] = True

        for m_idx, req_id in enumerate(self._request_ids):
            if req_id in scheduled_ids:
                continue
            for slot_idx in range(NUM_SLOTS):
                mask[NUM_READ_ACTIONS + m_idx * NUM_SLOTS + slot_idx] = True

        if not mask.any():
            mask[0] = True
        return mask

    def get_grader_score(self) -> Dict:
        return self.core_env.get_grader_score()

    # ------------------------------------------------------------------ encoding

    def _encode_obs(self, obs_dict: dict) -> np.ndarray:
        vec    = np.zeros(OBS_DIM, dtype=np.float32)
        offset = 0

        # 1. Availability matrix
        for p_idx, p in enumerate(obs_dict.get("participants", [])[:NUM_PARTICIPANTS]):
            for slot in p.get("availability", []):
                s = iso_to_slot_index(slot.get("start", ""))
                if 0 <= s < NUM_SLOTS:
                    vec[offset + p_idx * NUM_SLOTS + s] = 1.0
        offset += AVAIL_DIM

        # 2. Scheduled slots matrix
        for m in obs_dict.get("scheduled_meetings", []):
            s = iso_to_slot_index(m.get("time", ""))
            if 0 <= s < NUM_SLOTS:
                for pid in m.get("participants", []):
                    p_idx = PARTICIPANT_IDS.index(pid) if pid in PARTICIPANT_IDS else -1
                    if 0 <= p_idx < NUM_PARTICIPANTS:
                        vec[offset + p_idx * NUM_SLOTS + s] = 1.0
        offset += SCHED_DIM

        # 3. Profile read mask
        profiles_read = obs_dict.get("profiles_read", {})
        for i, pid in enumerate(PARTICIPANT_IDS):
            if pid in profiles_read:
                vec[offset + i] = 1.0
        offset += PROFILE_MASK_DIM

        # 4. Constraint features (only present when profile read)
        for i, pid in enumerate(PARTICIPANT_IDS):
            if pid in profiles_read:
                prof    = profiles_read[pid]
                prefs   = prof.get("preferred_times", []) if isinstance(prof, dict) else getattr(prof, "preferred_times", [])
                fatigue = prof.get("fatigue_penalty",  0.0) if isinstance(prof, dict) else getattr(prof, "fatigue_penalty",  0.0)
                for pref in (prefs if isinstance(prefs, list) else []):
                    if pref in PREF_MAP:
                        vec[offset + i * 4 + PREF_MAP[pref]] = 1.0
                vec[offset + i * 4 + 3] = float(fatigue)
        offset += CONSTRAINT_DIM

        # 5. Request completion flags
        scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}
        for i, req_id in enumerate(self._request_ids[:MAX_MEETINGS]):
            if req_id in scheduled_ids:
                vec[offset + i] = 1.0
        offset += REQUEST_DONE_DIM

        # 6. Step progress
        vec[offset] = obs_dict.get("step_count", 0) / self.core_env.max_steps
        return vec

    def _decode_action(self, action: int) -> Optional[dict]:
        if not (0 <= action < ACTION_DIM):
            return None
        if action < NUM_READ_ACTIONS:
            return {"action_type": "read_profile", "participant_id": PARTICIPANT_IDS[action]}
        idx         = action - NUM_READ_ACTIONS
        meeting_idx = idx // NUM_SLOTS
        slot_idx    = idx % NUM_SLOTS
        if meeting_idx >= len(self._request_ids):
            return None
        return {
            "action_type":   "schedule_meeting",
            "meeting_id":    self._request_ids[meeting_idx],
            "proposed_time": slot_index_to_iso(slot_idx),
        }


class CurriculumSchedulrXEnv(SchedulrXGymEnv):
    """
    Auto-advances difficulty when recent success rate exceeds threshold.
    Useful for PPO / DQN curriculum training pipelines.
    """

    LEVELS = ["easy", "medium", "hard"]

    def __init__(self, advance_threshold: float = 0.7, render_mode=None):
        super().__init__(task_name="easy", render_mode=render_mode)
        self.advance_threshold  = advance_threshold
        self.current_level      = 0
        self.episode_scores:    list = []
        self.episodes_per_eval  = 20

    def reset(self, seed=None, options=None):
        if len(self.episode_scores) >= self.episodes_per_eval:
            avg = sum(self.episode_scores[-self.episodes_per_eval:]) / self.episodes_per_eval
            if avg >= self.advance_threshold and self.current_level < len(self.LEVELS) - 1:
                self.current_level += 1
                self.episode_scores = []
        self.task_name = self.LEVELS[self.current_level]
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if done:
            self.episode_scores.append(self.get_grader_score()["score"])
        return obs, reward, done, truncated, info

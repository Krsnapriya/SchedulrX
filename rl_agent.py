"""
SchedulrX Heuristic RL Agent
==============================
A lightweight, deployable agent that uses the formal RL observation encoding
and action masking to make intelligent scheduling decisions.

This runs on HF Space (no SB3 dependency needed). It demonstrates:
- Proper state → action mapping via the Gymnasium interface
- Action masking (no invalid moves)
- Explore-then-exploit strategy (read profiles first, then schedule)
- Constraint-aware scheduling (uses discovered preferences)

For a trained PPO model, use train_rl.py locally and load with SB3.
"""

import numpy as np
from typing import Dict, List, Optional
from gym_env import (
    SchedulrXGymEnv,
    NUM_PARTICIPANTS, NUM_SLOTS, NUM_READ_ACTIONS,
    MAX_MEETINGS, PARTICIPANT_IDS, SLOT_HOURS,
    slot_index_to_iso, OBS_DIM, ACTION_DIM,
    AVAIL_DIM, SCHED_DIM, PROFILE_MASK_DIM, CONSTRAINT_DIM,
)
from env import SchedulrXEnv


class HeuristicRLAgent:
    """
    Smart heuristic agent built on the formal RL observation/action space.
    
    Strategy:
    Phase 1 (Explore): Read all participant profiles to discover hidden constraints.
    Phase 2 (Exploit): Schedule meetings in optimal slots using discovered info.
    
    Slot scoring uses:
    - Availability overlap (hard constraint)
    - Preference alignment (soft constraint from profiles)
    - Fatigue avoidance (back-to-back penalty)
    - Priority weighting (high-priority meetings first)
    """
    
    def __init__(self):
        self.name = "HeuristicRL-v1"
        self._failed_meetings = set()  # Track meetings we know can't be scheduled
    
    def select_action(self, obs: np.ndarray, mask: np.ndarray, env: SchedulrXGymEnv) -> int:
        """Select action using explore-then-exploit with action masking."""
        
        # Parse observation vector
        profiles_read = self._get_profiles_read(obs)
        requests_done = self._get_requests_done(obs)
        step_progress = obs[-1]
        
        # Phase 1: Explore — read unread profiles first
        unread_actions = []
        for i in range(NUM_READ_ACTIONS):
            if mask[i] and not profiles_read[i]:
                unread_actions.append(i)
        
        # Only read profiles relevant to pending meetings
        if unread_actions and step_progress < 0.3:
            # Prioritize participants involved in unscheduled meetings
            pending_participants = self._get_pending_participants(env)
            for a in unread_actions:
                pid = PARTICIPANT_IDS[a]
                if pid in pending_participants:
                    return a
            # Fallback: read any unread
            return unread_actions[0]
        
        # Phase 2: Exploit — schedule meetings by priority
        best_action = -1
        best_score = -999.0
        
        # Get meeting durations to check feasibility
        obs_dict = env.core_env._get_observation().model_dump()
        requests = obs_dict.get("requests", [])
        
        for m_idx in range(min(MAX_MEETINGS, len(env._request_ids))):
            if requests_done[m_idx]:
                continue
            
            # Skip meetings we've already failed on
            req_id = env._request_ids[m_idx] if m_idx < len(env._request_ids) else None
            if req_id in self._failed_meetings:
                continue
            
            # Check duration feasibility (slots are 60-min windows)
            if m_idx < len(requests):
                duration = requests[m_idx].get("duration_minutes", 60)
                if duration > 60:
                    # This meeting can't fit in any single slot — mark and skip
                    self._failed_meetings.add(req_id)
                    continue
        
        for m_idx in range(min(MAX_MEETINGS, len(env._request_ids))):
            if requests_done[m_idx]:
                continue
            req_id = env._request_ids[m_idx] if m_idx < len(env._request_ids) else None
            if req_id in self._failed_meetings:
                continue
            
            for slot_idx in range(NUM_SLOTS):
                action_idx = NUM_READ_ACTIONS + m_idx * NUM_SLOTS + slot_idx
                if action_idx >= ACTION_DIM or not mask[action_idx]:
                    continue
                
                score = self._score_slot(obs, env, m_idx, slot_idx)
                if score > best_score:
                    best_score = score
                    best_action = action_idx
        
        if best_action >= 0:
            return best_action
        
        # No valid scheduling actions and no useful reads — signal done
        # Return a read_profile action if any exist (low cost), otherwise fallback
        for i in range(NUM_READ_ACTIONS):
            if mask[i]:
                return i
        
        # Absolute fallback
        valid = np.where(mask)[0]
        return int(valid[0]) if len(valid) > 0 else 0
    
    
    def _score_slot(self, obs: np.ndarray, env: SchedulrXGymEnv, m_idx: int, slot_idx: int) -> float:
        """Score a time slot for a meeting based on constraints."""
        score = 0.0
        
        # Get the request's participants
        obs_dict = env.core_env._get_observation().model_dump()
        requests = obs_dict.get("requests", [])
        if m_idx >= len(requests):
            return -999.0
        
        req = requests[m_idx]
        participants = req.get("participants", [])
        priority = req.get("priority", 5) / 10.0
        
        # Base priority bonus
        score += priority * 2.0
        
        # Check availability for all participants
        for pid in participants:
            p_idx = PARTICIPANT_IDS.index(pid) if pid in PARTICIPANT_IDS else -1
            if p_idx < 0:
                continue
            
            avail_idx = p_idx * NUM_SLOTS + slot_idx
            if avail_idx < AVAIL_DIM and obs[avail_idx] > 0.5:
                score += 1.0  # Available
            else:
                score -= 5.0  # Not available — hard penalty
            
            # Check if slot already scheduled
            sched_idx = AVAIL_DIM + p_idx * NUM_SLOTS + slot_idx
            if sched_idx < AVAIL_DIM + SCHED_DIM and obs[sched_idx] > 0.5:
                score -= 10.0  # Conflict
        
        # Constraint-aware scoring (if profiles are read)
        profiles_read = obs_dict.get("profiles_read", {})
        constraint_offset = AVAIL_DIM + SCHED_DIM + PROFILE_MASK_DIM
        
        for pid in participants:
            p_idx = PARTICIPANT_IDS.index(pid) if pid in PARTICIPANT_IDS else -1
            if p_idx < 0 or pid not in profiles_read:
                continue
            
            # Get constraint features from observation
            c_base = constraint_offset + p_idx * 4
            pref_morning = obs[c_base] if c_base < OBS_DIM else 0
            pref_afternoon = obs[c_base + 1] if c_base + 1 < OBS_DIM else 0
            pref_evening = obs[c_base + 2] if c_base + 2 < OBS_DIM else 0
            fatigue = obs[c_base + 3] if c_base + 3 < OBS_DIM else 0
            
            # Match slot hour to preference
            hour_idx = slot_idx % 4
            slot_hour = SLOT_HOURS[hour_idx]
            
            if slot_hour < 12 and pref_morning > 0.5:
                score += 0.5  # Preference match
            elif 12 <= slot_hour < 16 and pref_afternoon > 0.5:
                score += 0.5
            elif slot_hour >= 16 and pref_evening > 0.5:
                score += 0.5
            
            # Fatigue penalty for participants with many meetings
            if fatigue > 0.3:
                score -= fatigue * 0.3
        
        return score
    
    def _get_profiles_read(self, obs: np.ndarray) -> np.ndarray:
        offset = AVAIL_DIM + SCHED_DIM
        return obs[offset:offset + PROFILE_MASK_DIM]
    
    def _get_requests_done(self, obs: np.ndarray) -> np.ndarray:
        offset = AVAIL_DIM + SCHED_DIM + PROFILE_MASK_DIM + CONSTRAINT_DIM
        return obs[offset:offset + MAX_MEETINGS]
    
    def _get_pending_participants(self, env: SchedulrXGymEnv) -> set:
        """Get participant IDs involved in unscheduled meetings."""
        obs_dict = env.core_env._get_observation().model_dump()
        scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}
        pids = set()
        for r in obs_dict.get("requests", []):
            if r["id"] not in scheduled_ids:
                pids.update(r["participants"])
        return pids


def run_heuristic_rl(task_name: str = "hard", n_episodes: int = 1) -> Dict:
    """
    Run the heuristic RL agent on SchedulrX and return results.
    This is the deployment-ready inference function used by the API.
    """
    agent = HeuristicRLAgent()
    env = SchedulrXGymEnv(task_name=task_name)
    
    all_scores = []
    all_rewards = []
    all_steps = []
    all_trajectories = []
    
    for ep in range(n_episodes):
        agent._failed_meetings = set()  # Reset per episode
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        trajectory = []
        
        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(obs, mask, env)
            
            decoded = env._decode_action(action)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            
            trajectory.append({
                "step": step,
                "action": decoded,
                "reward": round(reward, 4),
                "cumulative_reward": round(total_reward, 4),
            })
            
            # Early termination: all feasible meetings scheduled
            obs_dict = env.core_env._get_observation().model_dump()
            scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}
            all_req_ids = set(env._request_ids)
            feasible_remaining = all_req_ids - scheduled_ids - agent._failed_meetings
            if not feasible_remaining:
                break
        
        score = env.get_grader_score()
        all_scores.append(score["score"])
        all_rewards.append(total_reward)
        all_steps.append(step)
        all_trajectories.append(trajectory)
    
    result = {
        "agent": agent.name,
        "task": task_name,
        "episodes": n_episodes,
        "mean_score": round(float(np.mean(all_scores)), 3),
        "mean_reward": round(float(np.mean(all_rewards)), 3),
        "mean_steps": round(float(np.mean(all_steps)), 1),
        "grader": env.get_grader_score(),
    }
    
    if n_episodes == 1:
        result["trajectory"] = all_trajectories[0]
    
    return result


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        result = run_heuristic_rl(task_name=task, n_episodes=5)
        print(f"\n{task.upper()}: score={result['mean_score']}, reward={result['mean_reward']}, steps={result['mean_steps']}")

"""
SchedulrX Baseline Agent
========================
Uses OpenAI-compatible client via API_BASE_URL / MODEL_NAME / HF_TOKEN.
Falls back to zero-dependency heuristic if credentials are absent.
"""

import os
import json

from env import SchedulrXEnv
from models.schemas import Action

API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")


# --------------------------------------------------------------------------- heuristic

def _heuristic_baseline(env: SchedulrXEnv, task_name: str, max_steps: int = 80) -> dict:
    """
    Phase 1: read every relevant participant profile.
    Phase 2: greedily schedule each pending request at the first valid slot.
    """
    obs    = env.reset(task_name)
    total_reward = 0.0
    step   = 0
    done   = False
    profiles_read: set = set()

    while not done and step < max_steps:
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
        scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}
        pending = [r for r in obs_dict["requests"] if r["id"] not in scheduled_ids]

        if not pending:
            break

        needed_pids = {pid for r in pending for pid in r["participants"]}
        unread = [pid for pid in needed_pids if pid not in profiles_read]

        if unread:
            action = Action(action_type="read_profile", participant_id=unread[0])
            profiles_read.add(unread[0])
        else:
            req = pending[0]
            slot_time = None
            for p in obs_dict["participants"]:
                if p["id"] in req["participants"] and p.get("availability"):
                    slot_time = p["availability"][0]["start"]
                    break
            if slot_time is None:
                break
            action = Action(
                action_type="schedule_meeting",
                meeting_id=req["id"],
                proposed_time=slot_time,
            )

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    score_data = env.get_grader_score()
    return {
        "task":        task_name,
        "agent":       "heuristic",
        "final_score": score_data["score"],
        "total_reward": round(total_reward, 3),
        "steps_used":  step,
        "completed":   score_data["completed"],
        "total":       score_data["total"],
        "components":  score_data.get("components", {}),
    }


# --------------------------------------------------------------------------- llm

def _llm_baseline(env: SchedulrXEnv, task_name: str, max_steps: int = 60) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    SYSTEM = (
        "You are a meeting scheduler agent. Each turn, choose exactly ONE action "
        "and respond with a JSON object only (no markdown).\n\n"
        "Actions:\n"
        '  {"action_type": "read_profile", "participant_id": "<p1..p5>"}\n'
        '  {"action_type": "schedule_meeting", "meeting_id": "<id>", "proposed_time": "<ISO8601>"}\n\n'
        "Strategy:\n"
        "1. Read the profile of every participant in pending meetings FIRST.\n"
        "   Profiles reveal avoid_days — scheduling on those days always fails.\n"
        "2. Then schedule at times that fit within availability blocks "
        "   and avoid each participant's forbidden days.\n"
        "3. Meeting must fit ENTIRELY within one availability block."
    )

    obs  = env.reset(task_name)
    total_reward = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
        scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}

        compact = {
            "pending_meetings": [
                r for r in obs_dict["requests"] if r["id"] not in scheduled_ids
            ],
            "scheduled": list(scheduled_ids),
            "participants": [
                {
                    "id": p["id"], "name": p["name"], "timezone": p["timezone"],
                    "availability": p.get("availability", [])[:6],
                }
                for p in obs_dict["participants"]
            ],
            "known_profiles": {
                k: v for k, v in obs_dict.get("profiles_read", {}).items()
            },
            "step": step,
        }

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": json.dumps(compact, default=str)},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        content = (resp.choices[0].message.content or "").strip()
        content = content.lstrip("```json").lstrip("```").rstrip("```").strip()
        action  = Action(**json.loads(content))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    score_data = env.get_grader_score()
    return {
        "task":        task_name,
        "agent":       MODEL_NAME,
        "final_score": score_data["score"],
        "total_reward": round(total_reward, 3),
        "steps_used":  step,
        "completed":   score_data["completed"],
        "total":       score_data["total"],
        "components":  score_data.get("components", {}),
    }


# --------------------------------------------------------------------------- public

def run_openai_baseline(env: SchedulrXEnv, task_name: str = "hard", max_steps: int = 60) -> dict:
    if API_KEY and API_BASE_URL and MODEL_NAME:
        try:
            return _llm_baseline(env, task_name, max_steps)
        except Exception as exc:
            result = _heuristic_baseline(env, task_name, max_steps)
            result["fallback_reason"] = str(exc)[:200]
            return result
    result = _heuristic_baseline(env, task_name, max_steps)
    result["note"] = "No API credentials — heuristic baseline used."
    return result


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        e = SchedulrXEnv()
        print(run_openai_baseline(e, task))

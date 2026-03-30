"""
SchedulrX OpenAI Baseline Agent
================================
FIX #14: Uses API_BASE_URL, MODEL_NAME, HF_TOKEN as mandated by the hackathon spec.
FIX #15: Removed response_format strict json_schema (fails with Optional/anyOf schemas).
FIX #16: Creates a fresh env per task call.

Run:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python baseline.py
"""

import os
import json
from openai import OpenAI
from models.schemas import Action
from env import SchedulrXEnv

# FIX #14: mandatory hackathon env vars
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM = """You are an expert meeting scheduler. Choose ONE action per turn.

Respond with valid JSON only — no markdown, no explanation.

Actions:
  {"action_type": "read_profile", "participant_id": "<id>"}
  {"action_type": "schedule_meeting", "meeting_id": "<id>", "proposed_time": "<ISO datetime>"}

Strategy: read profiles for all participants in pending meetings first, then schedule."""


def run_openai_baseline(env: SchedulrXEnv, task_name: str = "easy", max_steps: int = 30) -> dict:
    # FIX #16: always reset a fresh environment
    obs = env.reset(task_name=task_name, seed=42)
    total_reward = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        obs_dict = obs.model_dump(mode="json") if hasattr(obs, "model_dump") else obs

        try:
            # FIX #15: plain json_object mode instead of strict json_schema
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(obs_dict, default=str)},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            action_dict = json.loads(response.choices[0].message.content or "{}")
            action = Action(**action_dict)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
        except Exception as e:
            print(f"[baseline] error at step {step}: {e}")
            break

    result = env.get_grader_score()
    return {
        "task": task_name,
        "model": MODEL_NAME,
        "final_score": result["score"],
        "total_reward": round(total_reward, 3),
        "steps_used": step,
        "grader": result,
    }


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        env = SchedulrXEnv()   # FIX #16: fresh env per task
        r = run_openai_baseline(env, task)
        print(f"{task}: score={r['final_score']}  reward={r['total_reward']}  steps={r['steps_used']}")
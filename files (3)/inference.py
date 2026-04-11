"""
SchedulrX Inference Script
===========================
Mandatory stdout format:
  [START] task=<name> env=schedulrx model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os
import json
import re
import sys

from openai import OpenAI
from env import SchedulrXEnv
from models.schemas import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

BENCHMARK  = "schedulrx"
MAX_STEPS  = 30
TEMP       = 0.1
MAX_TOKENS = 300

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM = (
    "You are a meeting scheduler. Each turn respond with ONE JSON action only.\n\n"
    "Actions:\n"
    '  {"action_type": "read_profile", "participant_id": "<p1..p5>"}\n'
    '  {"action_type": "schedule_meeting", "meeting_id": "<id>", "proposed_time": "<ISO8601>"}\n\n'
    "Critical rules:\n"
    "- Read every participant's profile BEFORE scheduling their meetings.\n"
    "  Profiles reveal avoid_days. Scheduling on an avoided day always fails.\n"
    "- A meeting must fit entirely within one availability block.\n"
    "- Respond with raw JSON only. No markdown, no explanation."
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in: {text[:200]}")


def run_task(task_name: str):
    env  = SchedulrXEnv()
    obs  = env.reset(task_name)
    done = False
    step = 0
    rewards: list = []
    action_dict: dict = {}

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    while not done and step < MAX_STEPS:
        step      += 1
        reward     = 0.0
        error_msg  = "null"

        try:
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
            scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}

            compact = {
                "pending": [r for r in obs_dict["requests"] if r["id"] not in scheduled_ids],
                "scheduled": list(scheduled_ids),
                "participants": [
                    {
                        "id": p["id"], "name": p["name"], "timezone": p["timezone"],
                        "availability": p.get("availability", [])[:6],
                    }
                    for p in obs_dict["participants"]
                ],
                "known_profiles": {
                    k: {"avoid_days": v.get("avoid_days", []) if isinstance(v, dict) else [],
                        "preferred_times": v.get("preferred_times", []) if isinstance(v, dict) else []}
                    for k, v in obs_dict.get("profiles_read", {}).items()
                },
                "step": step,
            }

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": json.dumps(compact, default=str)},
                ],
                temperature=TEMP,
                max_tokens=MAX_TOKENS,
            )

            raw = resp.choices[0].message.content or ""
            action_dict = _extract_json(raw)
            action      = Action(**action_dict)
            obs, reward, done, info = env.step(action)

        except Exception as exc:
            error_msg = str(exc).replace("\n", " ")[:200]
            reward    = -0.1
            if "avoids" not in error_msg and "availability" not in error_msg:
                done = True

        a_str = json.dumps(action_dict) if action_dict else "null"
        print(
            f"[STEP] step={step} action={a_str} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error={error_msg}",
            flush=True,
        )
        rewards.append(reward)

    grader  = env.get_grader_score()
    score   = grader.get("score", 0.0)
    success = "true" if score >= 0.5 else "false"
    r_str   = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={success} steps={step} score={score:.2f} rewards={r_str}",
        flush=True,
    )


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as exc:
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)

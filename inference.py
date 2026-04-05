"""
SchedulrX Inference Script
===========================
MANDATORY env vars:
  API_BASE_URL   — LLM endpoint (default required)
  MODEL_NAME     — model identifier (default required)
  HF_TOKEN       — API key (mandatory, no default)

STDOUT FORMAT (strict — any deviation breaks scoring):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import json
import re
import sys
from openai import OpenAI
from server.env import SchedulrXEnv
from server.models.schemas import Action

# --- Mandatory env vars ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

BENCHMARK   = "schedulrx"
TEMPERATURE = 0.2
MAX_TOKENS  = 512

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert meeting scheduler agent. Choose ONE action per turn.

Available actions (respond with valid JSON only, no markdown):

1. Read a participant's hidden constraints:
   {"action_type": "read_profile", "participant_id": "<id>"}

2. Schedule a meeting:
   {"action_type": "schedule_meeting", "meeting_id": "<id>", "proposed_time": "<ISO datetime>"}

Strategy:
- Read profiles for all participants in pending meetings FIRST.
- Then schedule meetings at times that fit entirely within an availability slot.
- Meeting duration must fit within the slot window.

Respond with ONLY a JSON object."""


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from: {text[:200]}")


def run_task(task_name: str):
    env = SchedulrXEnv()
    MAX_STEPS = env.max_steps

    # [START] marker
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    obs = env.reset(task_name=task_name, seed=42)

    rewards_list = []
    total_steps = 0
    done = False
    action_dict = {}

    while not done and total_steps < MAX_STEPS:
        total_steps += 1
        error_msg = "null"
        reward = 0.0

        try:
            obs_dict = obs.model_dump(mode="json") if hasattr(obs, "model_dump") else obs

            # Compact observation
            scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}
            compact_obs = {
                "step": obs_dict.get("step_count", total_steps),
                "participants": [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "timezone": p["timezone"],
                        "availability": p.get("availability", []),
                    }
                    for p in obs_dict.get("participants", [])
                ],
                "pending_meetings": [
                    r for r in obs_dict.get("requests", [])
                    if r["id"] not in scheduled_ids
                ],
                "scheduled": obs_dict.get("scheduled_meetings", []),
                "known_constraints": {
                    k: {
                        "preferred": v.get("preferred_times", []) if isinstance(v, dict) else [],
                        "avoid_days": v.get("avoid_days", []) if isinstance(v, dict) else [],
                        "fatigue": v.get("fatigue_penalty", 0) if isinstance(v, dict) else 0,
                    }
                    for k, v in obs_dict.get("profiles_read", {}).items()
                },
            }

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(compact_obs, default=str)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            raw_content = response.choices[0].message.content or ""
            action_dict = extract_json(raw_content)
            action = Action(**action_dict)
            obs, reward, done, info = env.step(action)

        except Exception as e:
            error_msg = str(e).replace("\n", " ").replace("\r", "")[:200]
            if "Could not extract JSON" in error_msg or "validation error" in error_msg.lower():
                reward = -0.1
            else:
                reward = 0.0
                done = True

        a_str = json.dumps(action_dict) if action_dict else "null"
        # [STEP] marker
        print(
            f"[STEP] step={total_steps} action={a_str} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error={error_msg}",
            flush=True,
        )
        rewards_list.append(reward)

    # [END] marker
    grader_res = env.get_grader_score()
    score = grader_res.get("score", 0.0)
    success_str = "true" if score >= 0.5 else "false"
    r_list_str = ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"
    
    # STRICT: [END] success=<bool> steps=<n> rewards=<r1,r2,...>
    print(
        f"[END] success={success_str} steps={total_steps} rewards={r_list_str}",
        flush=True,
    )


if __name__ == "__main__":
    # Allow specifying a single task via env var
    single_task = os.getenv("TASK_NAME")
    if single_task:
        tasks_to_run = [single_task]
    else:
        tasks_to_run = ["easy", "medium", "hard"]

    for task in tasks_to_run:
        try:
            run_task(task)
        except Exception as e:
            # Emergency [END] so parser doesn't break
            print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=0.00", flush=True)
            print(f"[ERROR] {e}", file=sys.stderr, flush=True)

"""
SchedulrX Inference Script
===========================
MANDATORY env vars injected by validator:
  API_BASE_URL  — LiteLLM proxy endpoint
  HF_TOKEN      — proxy key
  MODEL_NAME    — model identifier (optional, defaults to gpt-4o-mini)

STDOUT FORMAT:
  [START] task=<name> env=<benchmark> model=<model>
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

# MANDATORY — no fallbacks, must use injected proxy vars
API_BASE_URL = os.environ["API_BASE_URL"]
HF_TOKEN      = os.environ["HF_TOKEN"]
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")

BENCHMARK   = "schedulrx"
TEMPERATURE = 0.2
MAX_TOKENS  = 512

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert meeting scheduler agent. Choose ONE action per turn.

Respond with valid JSON only — no markdown, no explanation.

Actions:
  {"action_type": "read_profile", "participant_id": "<id>"}
  {"action_type": "schedule_meeting", "meeting_id": "<id>", "proposed_time": "<ISO datetime>"}
  {"action_type": "accept_proposal", "proposal_id": "<id>"}
  {"action_type": "reschedule_meeting", "meeting_id": "<id>", "proposed_time": "<ISO datetime>"}

Strategy:
1. ALWAYS read profiles for every participant in a pending meeting BEFORE scheduling to see hidden availability.
2. Check `depends_on` in meeting requests. If Meeting B depends on Meeting A, ensure Meeting A is scheduled for an EARLIER time than Meeting B.
3. Check `deadline_hours`. Meetings must be scheduled before this relative deadline.
4. If a participant offers a `counter_proposals`, use `accept_proposal` to quickly resolve conflicts.
5. If a meeting appears in `cancelled_meetings`, use `reschedule_meeting` to find a new slot.
6. Only schedule meetings within known availability slots."""


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

    # [START] printed FIRST before anything can crash
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    obs = env.reset(task_name=task_name, seed=42)

    rewards_list = []
    total_steps  = 0
    done         = False
    action_dict  = {}

    while not done and total_steps < MAX_STEPS:
        total_steps += 1
        error_msg = "null"
        reward    = 0.0

        try:
            obs_dict = obs.model_dump(mode="json") if hasattr(obs, "model_dump") else obs
            scheduled_ids = {m["meeting_id"] for m in obs_dict.get("scheduled_meetings", [])}

            compact = {
                "step": obs_dict.get("step_count", total_steps),
                "participants": obs_dict.get("participants", []),
                "pending_meetings": [
                    r for r in obs_dict.get("requests", [])
                    if r["id"] not in scheduled_ids
                ],
                "scheduled": obs_dict.get("scheduled_meetings", []),
                "cancelled": obs_dict.get("cancelled_meetings", []),
                "counter_proposals": obs_dict.get("counter_proposals", []),
                "known_constraints": obs_dict.get("profiles_read", {}),
            }

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(compact, default=str)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            raw = response.choices[0].message.content or ""
            action_dict = extract_json(raw)
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
        print(
            f"[STEP] step={total_steps} action={a_str} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error={error_msg}",
            flush=True,
        )
        rewards_list.append(reward)

    score_data = env.get_grader_score()
    score      = score_data.get("score", 0.0)
    success    = "true" if score >= 0.5 else "false"
    r_list_str = ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"
    print(
        f"[END] success={success} steps={total_steps} score={score:.2f} rewards={r_list_str}",
        flush=True,
    )


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
            print(f"[FATAL] {e}", file=sys.stderr, flush=True)

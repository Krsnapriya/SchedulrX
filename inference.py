"""
SchedulrX Inference Script
===========================
MANDATORY env vars:
  API_BASE_URL  — LLM endpoint (default: https://api.openai.com/v1)
  MODEL_NAME    — model identifier (default: gpt-4o-mini)
  HF_TOKEN      — API key (mandatory, no default)

STDOUT FORMAT:
  [START] task=<task_name> env=schedulrx model=<model_name>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import os
import json
import sys
import requests
from openai import OpenAI

# ====================== EXACT ENV VAR HANDLING ======================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ====================== BENCHMARK CONFIG ======================
ENV_BASE_URL = "https://krsnapriya-meeting-scheduler-openenv.hf.space"
BENCHMARK_NAME = "schedulrx"
TASKS = ["easy", "medium", "hard"]


def run_task(task_name: str):
    """Run a single task episode and emit [START]/[STEP]/[END] to stdout."""

    # ====================== START ======================
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}")

    try:
        # Reset
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset", params={"task_name": task_name}, timeout=30
        )
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
        session_id = reset_data["session_id"]
        obs = reset_data["observation"]
    except Exception as e:
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={e}")
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00")
        return

    reward_list = []
    step_num = 0
    done = False
    last_error = None

    while not done and step_num < 30:
        step_num += 1

        # Build LLM prompt
        prompt = f"""You are an expert meeting scheduler. Current state:
{json.dumps(obs, default=str, indent=2)}

Return ONLY valid JSON for one action. No extra text.

Examples:
{{"action_type": "read_profile", "participant_id": "p1"}}
{{"action_type": "schedule_meeting", "meeting_id": "r1", "proposed_time": "2026-04-12T10:00:00+00:00"}}

Valid action_type: read_profile, schedule_meeting, accept_proposal, reschedule_meeting"""

        try:
            llm_resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            content = llm_resp.choices[0].message.content.strip()

            # Clean markdown if model adds it
            if content.startswith("```"):
                content = content.split("```")[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()

            action_dict = json.loads(content)

            # Step
            step_payload = {"session_id": session_id, "action": action_dict}
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step", json=step_payload, timeout=30
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            obs = step_data["observation"]
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)

            reward_list.append(reward)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

        except Exception as e:
            last_error = str(e).replace("\n", " ")[:200]
            reward_list.append(0.0)
            print(
                f"[STEP] step={step_num} action=null "
                f"reward=0.00 done=false error={last_error}"
            )
            break

    # ====================== SCORE ======================
    try:
        grader_resp = requests.get(
            f"{ENV_BASE_URL}/grader",
            params={"session_id": session_id},
            timeout=15,
        )
        grader_resp.raise_for_status()
        score = grader_resp.json().get("score", 0.0)
    except Exception:
        score = 0.0

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # ====================== END ======================
    success = done and last_error is None
    rewards_str = ",".join(f"{r:.2f}" for r in reward_list) if reward_list else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={score:.2f} rewards={rewards_str}"
    )


if __name__ == "__main__":
    # Run all 3 tasks sequentially (or specific task from CLI)
    if len(sys.argv) > 1:
        tasks_to_run = [sys.argv[1]]
    else:
        tasks_to_run = TASKS

    for task in tasks_to_run:
        try:
            run_task(task)
        except Exception as e:
            print(f"[START] task={task} env={BENCHMARK_NAME} model={MODEL_NAME}")
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
            print(f"[FATAL] {e}", file=sys.stderr, flush=True)

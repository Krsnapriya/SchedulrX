"""
SchedulrX Baseline Inference Script
====================================
Runs an LLM agent against all 3 SchedulrX tasks (easy, medium, hard)
and logs results in the mandatory [START]/[STEP]/[END] format.

Required environment variables:
  OPENAI_API_KEY  - API key for LLM calls
  API_BASE_URL    - LLM API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME      - Model identifier (default: gpt-4o-mini)
"""

import os
import sys
import asyncio
import json
from openai import OpenAI
import httpx

# --- Mandatory env vars per spec ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    print("WARNING: OPENAI_API_KEY not found. Automated validation may fail during LLM calls.")
    OPENAI_API_KEY = "sk-placeholder-for-validator"

client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

# Point to the live HF Space
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://krsnapriya-meeting-scheduler-openenv.hf.space"
)

BENCHMARK = "schedulrx"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = {"easy": 30, "medium": 30, "hard": 40}
SUCCESS_THRESHOLDS = {"easy": 0.80, "medium": 0.60, "hard": 0.40}


# --- Structured logging (exact spec format) ---
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    action_str = json.dumps(action, separators=(",", ":")) if action else "null"
    error_str = json.dumps(str(error)) if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_str}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={json.dumps(rewards)}",
        flush=True,
    )


def build_prompt(obs):
    """Build the agent prompt from the current observation."""
    return f"""You are an AI agent solving a meeting scheduling environment.

Rules:
1. Participants' availabilities are hidden. Use 'read_profile' to discover them.
2. You have a limited Trust Budget. Check 'trust_scores' — do not read if budget is 0.
3. In hard mode, you MUST read P5's profile before scheduling meeting r1, or it will be rejected.
4. Meetings may have a 'depends_on' field — schedule the dependency first.
5. Check 'counter_proposals' — use 'accept_proposal' with the proposal_id if one exists.
6. Check 'cancelled_meetings' — use 'reschedule_meeting' for any cancelled meeting_id.
7. Availability windows span 9 AM to 6 PM local time for each participant across 5 weekdays.

Current Observation:
{json.dumps(obs, default=str)}

Respond with ONLY a JSON object:
{{"action_type": "read_profile|schedule_meeting|accept_proposal|reschedule_meeting", "participant_id": "optional", "meeting_id": "optional", "proposed_time": "optional ISO8601", "proposal_id": "optional"}}"""


async def run_task(task_name: str):
    """Run a single task and return (score, rewards, steps, success)."""
    max_steps = MAX_STEPS_PER_TASK[task_name]
    threshold = SUCCESS_THRESHOLDS[task_name]
    rewards = []
    steps_taken = 0
    score = 0.0
    last_error = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            # Reset
            reset_resp = await http.post(
                f"{ENV_BASE_URL}/reset", params={"task_name": task_name}
            )
            reset_data = reset_resp.json()
            session_id = reset_data["session_id"]
            obs = reset_data["observation"]

            for step in range(1, max_steps + 1):
                # LLM call
                try:
                    llm_resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a deterministic AI agent. Output raw JSON only."},
                            {"role": "user", "content": build_prompt(obs)},
                        ],
                        temperature=0.0,
                        max_tokens=300,
                    )
                    action_text = llm_resp.choices[0].message.content.strip()
                    # Strip markdown fences if present
                    if action_text.startswith("```"):
                        action_text = action_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    action_dict = json.loads(action_text)
                except Exception as e:
                    log_step(step=step, action=None, reward=0.0, done=False, error=e)
                    last_error = str(e)
                    steps_taken = step
                    break

                # Step
                step_resp = await http.post(
                    f"{ENV_BASE_URL}/step",
                    json={"session_id": session_id, "action": action_dict},
                )
                step_data = step_resp.json()

                obs = step_data["observation"]
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_dict, reward=reward, done=done, error=None)

                if done:
                    break

            # Get grader score
            grader_resp = await http.get(
                f"{ENV_BASE_URL}/grader", params={"session_id": session_id}
            )
            score = grader_resp.json().get("score", 0.0)

    except Exception as e:
        last_error = str(e)
        log_step(step=steps_taken + 1, action=None, reward=0.0, done=False, error=e)
        score = 0.0

    success = score >= threshold and last_error is None
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score, rewards, steps_taken, success


async def main():
    """Run baseline across all 3 tasks."""
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())

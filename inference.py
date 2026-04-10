import os
import json
import sys
import requests
from openai import OpenAI

# ====================== EXACT ENV VAR HANDLING THEY REQUIRE ======================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ====================== OUR BENCHMARK CONFIG ======================
ENV_BASE_URL = "https://krsnapriya-meeting-scheduler-openenv.hf.space"
BENCHMARK_NAME = "schedulrx"

# Task from command line (hackathon can pass it) or default to hard
task_name = sys.argv[1] if len(sys.argv) > 1 else "hard"

# ====================== START ======================
print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}")

# Reset
reset_resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_name": task_name})
reset_data = reset_resp.json()
session_id = reset_data["session_id"]
obs = reset_data["observation"]

rewards = []
step_num = 0
done = False
last_error = None

while not done:
    step_num += 1

    # Prompt LLM (expert scheduler)
    prompt = f"""You are an expert meeting scheduler. Current state:
{json.dumps(obs, default=str, indent=2)}

Return ONLY valid JSON for one action. No extra text.

Examples:
{{"action_type": "read_profile", "participant_id": "p1"}}
{{"action_type": "schedule_meeting", "meeting_id": "r1", "proposed_time": "2026-04-07T10:00:00+05:30"}}

Valid action_type: schedule_meeting, reschedule_meeting, cancel_meeting, read_profile, propose_alternative"""

    try:
        llm_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
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
        step_resp = requests.post(f"{ENV_BASE_URL}/step", json=step_payload)
        step_data = step_resp.json()

        obs = step_data["observation"]
        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", False)

        rewards.append(reward)
        action_str = json.dumps(action_dict, separators=(",", ":"))

        print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

    except Exception as e:
        last_error = str(e)
        print(f"[STEP] step={step_num} action=null reward=0.00 done=false error={last_error}")
        break

# ====================== END ======================
success = done and last_error is None
rewards_str = ",".join(f"{r:.2f}" for r in rewards)
print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}")

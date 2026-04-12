import os
import asyncio
import json
from openai import OpenAI
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
ENV_BASE_URL = 'http://localhost:7860'

TASK_NAME = "hard"
BENCHMARK = "schedulrx"
MAX_STEPS = 80

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done, error):
    action_str = json.dumps(action, separators=(",", ":")) if action else "null"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}")

SUCCESS_THRESHOLDS = {"easy": 0.70, "medium": 0.50, "hard": 0.30}

async def main():
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    rewards = []
    steps_taken = 0
    last_error = None

    try:
        # Reset
        async with httpx.AsyncClient() as http:
            reset_resp = await http.post(f"{ENV_BASE_URL}/reset", params={"task_name": TASK_NAME})
            reset_data = reset_resp.json()
            session_id = reset_data["session_id"]
            obs = reset_data["observation"]

            for step in range(1, MAX_STEPS + 1):
                # LLM call
                prompt = f"""You are a God-Tier RL Agent solving a POMDP meeting scheduling environment.
Your capabilities:
1. OVERCOME PARTIAL OBSERVABILITY: Participants' availabilities are hidden. You MUST use 'read_profile' to discover them.
2. MANAGE TRUST BUDGET: You only have a limited amount of honest reads. Do not query if 'trust_scores' for a participant is 0.
3. ADVERSARIAL ROBUSTNESS: In hard mode, critical stakeholders (like P5) will reject schedules if you haven't read their profile first.
4. DEPENDENCY CHAINING: Meetings may have a 'depends_on' field mapping to another meeting_id. You MUST schedule the prerequisite meeting first.
5. COUNTER-PROPOSALS: If a chosen time violates a soft constraint, a participant may offer an alternative. Check the 'counter_proposals' array and use 'accept_proposal' with the 'proposal_id' if appropriate.
6. STOCHASTIC CANCELLATIONS: Meetings may unexpectedly be cancelled by the environment. Check 'cancelled_meetings' and immediately 'reschedule_meeting' for that meeting_id.
7. ACTION FORMAT: Respond ONLY with valid JSON. Valid action_type: "read_profile", "schedule_meeting", "accept_proposal", "reschedule_meeting".

Current State:
{json.dumps(obs, default=str)}

Return ONLY JSON: {{"action_type": string, "participant_id": optional_string, "meeting_id": optional_string, "proposed_time": optional_string, "proposal_id": optional_string}}"""
                llm_resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": "You are a deterministic AI agent outputting raw JSON."},
                              {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300
                )
                action_text = llm_resp.choices[0].message.content.strip()
                if action_text.startswith("```json"):
                    action_text = action_text[7:-3]
                action_dict = json.loads(action_text)


                # Step
                step_payload = {"session_id": session_id, "action": action_dict}
                step_resp = await http.post(f"{ENV_BASE_URL}/step", json=step_payload)
                step_data = step_resp.json()

                obs = step_data["observation"]
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_dict, reward=reward, done=done, error=None)

                if done:
                    break

            # Get final score from grader
            grader_resp = await http.get(f"{ENV_BASE_URL}/grader", params={"session_id": session_id})
            grader_data = grader_resp.json()
            score = grader_data.get("score", 0.0)

    except Exception as e:
        last_error = str(e)
        log_step(step=steps_taken+1, action=None, reward=0.0, done=False, error=last_error)
        score = 0.0

    finally:
        SUCCESS_THRESHOLDS = {"easy": 0.80, "medium": 0.60, "hard": 0.50}
        threshold = SUCCESS_THRESHOLDS.get(TASK_NAME, 0.50)
        success_str = "true" if score >= threshold and last_error is None else "false"
        print(f"[END] success={success_str} steps={steps_taken} score={score:.2f} rewards={rewards}")
        import sys
        sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())

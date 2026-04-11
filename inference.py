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
ENV_BASE_URL = "https://krsnapriya-meeting-scheduler-openenv.hf.space"

TASK_NAME = "hard"
BENCHMARK = "schedulrx"
MAX_STEPS = 80

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done, error):
    action_str = json.dumps(action, separators=(",", ":")) if action else "null"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}")

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")

async def main():
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    rewards = []
    steps_taken = 0
    success = False
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
                prompt = f"""Expert meeting scheduler. State:\n{json.dumps(obs, default=str)}\nReturn ONLY JSON action."""
                llm_resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300
                )
                action_dict = json.loads(llm_resp.choices[0].message.content.strip())

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
                    success = True
                    break

    except Exception as e:
        last_error = str(e)
        log_step(step=steps_taken+1, action=None, reward=0.0, done=False, error=last_error)

    finally:
        log_end(success=success and last_error is None, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())

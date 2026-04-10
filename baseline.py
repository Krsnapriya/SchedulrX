import os, json
from env import SchedulrXEnv
from models.schemas import Action

API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

def _heuristic_baseline(env: SchedulrXEnv, task_name: str, max_steps=80):
    """Zero-dependency heuristic: read all profiles first, then greedily schedule."""
    obs = env.reset(task_name)
    total_reward = 0.0
    step = 0
    done = False
    participant_ids = [p.id for p in obs.participants]
    profiles_read = set()

    while not done and step < max_steps:
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
        pending = [r for r in obs_dict["requests"]
                   if r["id"] not in {m["meeting_id"] for m in obs_dict["scheduled_meetings"]}]
        if not pending:
            break

        # Phase 1: read profiles
        unread = [pid for pid in participant_ids if pid not in profiles_read]
        if unread:
            action = Action(action_type="read_profile", participant_id=unread[0])
            profiles_read.add(unread[0])
        else:
            # Phase 2: schedule first pending meeting at first valid slot
            req = pending[0]
            # find first slot where all participants are free
            first_slot = None
            for p in obs_dict["participants"]:
                if p["id"] in req["participants"] and p.get("availability"):
                    first_slot = p["availability"][0]["start"]
                    break
            if first_slot is None:
                break
            action = Action(action_type="schedule_meeting",
                            meeting_id=req["id"],
                            proposed_time=first_slot)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    final_score = env.get_grader_score()["score"]
    return {"task": task_name, "final_score": round(final_score, 3),
            "total_reward": round(total_reward, 3), "steps_used": step}


def run_openai_baseline(env: SchedulrXEnv, task_name: str = "hard", max_steps=80):
    """LLM baseline — falls back to heuristic if no API key is configured."""
    if not API_KEY or not API_BASE_URL:
        return _heuristic_baseline(env, task_name, max_steps)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        obs = env.reset(task_name)
        total_reward = 0.0
        step = 0
        done = False
        while not done and step < max_steps:
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user",
                           "content": f"You are a scheduler. State:\n{json.dumps(obs_dict, default=str)}\nRespond with ONE JSON action only."}],
                temperature=0.0, max_tokens=300,
            )
            action = Action(**json.loads(response.choices[0].message.content))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
        final_score = env.get_grader_score()["score"]
        return {"task": task_name, "final_score": round(final_score, 3),
                "total_reward": round(total_reward, 3), "steps_used": step}
    except Exception as e:
        return _heuristic_baseline(env, task_name, max_steps) | {"error": str(e)}


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        env = SchedulrXEnv()
        print(run_openai_baseline(env, task))
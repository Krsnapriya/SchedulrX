import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict
from env import SchedulrXEnv
from models.schemas import Action

app = FastAPI(title="SchedulrX OpenEnv API", version="1.0.0")

# FIX #11/#12: bounded session store with TTL
_sessions: Dict[str, Dict] = {}   # {session_id: {"env": env, "created": timestamp}}
SESSION_TTL = 3600        # 1 hour
MAX_SESSIONS = 200


def _get_env(session_id: str) -> SchedulrXEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return _sessions[session_id]["env"]


def _cleanup_sessions():
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v["created"] > SESSION_TTL]
    for k in expired:
        del _sessions[k]
    # Hard cap: evict oldest if over limit
    if len(_sessions) > MAX_SESSIONS:
        oldest = sorted(_sessions.items(), key=lambda x: x[1]["created"])
        for k, _ in oldest[:len(_sessions) - MAX_SESSIONS]:
            del _sessions[k]


class StepRequest(BaseModel):
    session_id: str
    action: Action


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "environment": "SchedulrX",
        "version": "1.0.0",
        "active_sessions": len(_sessions),
    }


@app.post("/reset")
async def reset(task_name: str = "easy", seed: int = 42):
    _cleanup_sessions()
    session_id = str(uuid4())
    env = SchedulrXEnv()
    obs = env.reset(task_name=task_name, seed=seed)
    _sessions[session_id] = {"env": env, "created": time.time()}
    return {"session_id": session_id, "observation": obs.model_dump(mode="json")}


@app.post("/step")
async def step(req: StepRequest):
    env = _get_env(req.session_id)
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(mode="json"),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def get_state(session_id: str):
    env = _get_env(session_id)
    return env.state()


@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": [
            {"name": "easy",   "description": "1 meeting, 2 participants, no hidden traps"},
            {"name": "medium", "description": "3 meetings, 4 participants, timezone conflicts"},
            {"name": "hard",   "description": "3 meetings, 5 participants, hidden constraints + fatigue"},
        ],
        "action_schema": Action.model_json_schema(),
    }


@app.get("/grader")
async def grader(session_id: str):
    env = _get_env(session_id)
    return env.get_grader_score()


@app.post("/baseline")
async def run_baseline(task_name: str = "easy"):
    """
    Run the LLM baseline agent.
    FIX #13: uses API_BASE_URL / MODEL_NAME / HF_TOKEN — the mandatory hackathon vars.
    """
    from baseline import run_openai_baseline
    env = SchedulrXEnv()
    result = run_openai_baseline(env, task_name)
    return result


@app.post("/rl-baseline")
async def rl_baseline(task_name: str = "easy"):
    """Run the heuristic RL agent (no external API key needed)."""
    from rl_agent import run_heuristic_rl
    return run_heuristic_rl(task_name=task_name, n_episodes=1)

def main():
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)

if __name__ == "__main__":
    main()

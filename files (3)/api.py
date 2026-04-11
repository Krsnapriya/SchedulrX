from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Optional
from uuid import uuid4

from env import SchedulrXEnv
from models.schemas import Action

app = FastAPI(
    title="SchedulrX OpenEnv API",
    description=(
        "POMDP meeting-scheduling environment for RL research. "
        "Agents must discover hidden participant constraints via read_profile "
        "before scheduling, creating a genuine information-gathering challenge."
    ),
    version="2.0.0",
)

sessions: Dict[str, SchedulrXEnv] = {}

TASK_METADATA = [
    {
        "name": "easy",
        "description": (
            "Schedule one 45-minute meeting between two participants (Bob, Dave) "
            "who have no avoid_day constraints. Validates basic env interaction."
        ),
        "participants": 2,
        "meetings": 1,
        "hidden_traps": 0,
        "grader": True,
        "grader_endpoint": "/grader",
        "difficulty": 1,
    },
    {
        "name": "medium",
        "description": (
            "Schedule 3 meetings across 4 participants over different timezones. "
            "Carol (p3) avoids Friday — agents that skip profile discovery will "
            "hit slot failures on that day."
        ),
        "participants": 4,
        "meetings": 3,
        "hidden_traps": 1,
        "grader": True,
        "grader_endpoint": "/grader",
        "difficulty": 2,
    },
    {
        "name": "hard",
        "description": (
            "Schedule 3 meetings (including a 90-minute offsite) across 5 participants "
            "spanning 5 timezones. Alice avoids weekends, Eve avoids Sunday/Monday, "
            "Carol avoids Friday. Greedy agents fail repeatedly; optimal policy requires "
            "reading all 3 at-risk profiles before any scheduling attempt."
        ),
        "participants": 5,
        "meetings": 3,
        "hidden_traps": 3,
        "grader": True,
        "grader_endpoint": "/grader",
        "difficulty": 3,
    },
]

TASK_NAMES = [t["name"] for t in TASK_METADATA]


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "schedulrx",
        "version": "2.0.0",
        "tasks": TASK_NAMES,
    }


@app.post("/reset")
async def reset(task_name: str = "easy"):
    if task_name not in TASK_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid: {TASK_NAMES}",
        )
    session_id = str(uuid4())
    env = SchedulrXEnv()
    sessions[session_id] = env
    obs = env.reset(task_name)
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
async def step(req_body: dict):
    session_id = req_body.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id. Call /reset first.")
    action_data = req_body.get("action", {})
    action = Action(**action_data)
    env = sessions[session_id]
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    return sessions[session_id].state()


@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": TASK_METADATA,
        "task_names": TASK_NAMES,
        "action_schema": Action.model_json_schema(),
    }


@app.get("/grader")
async def grader(session_id: Optional[str] = None, task_name: Optional[str] = None):
    """
    Stateful:  GET /grader?session_id=<id>      — scores active session
    Stateless: GET /grader?task_name=easy|medium|hard — runs heuristic, returns score
    The stateless path is what the Phase 2 validator uses.
    """
    if session_id and session_id in sessions:
        return sessions[session_id].get_grader_score()

    task = task_name if task_name in TASK_NAMES else "easy"
    env = SchedulrXEnv()
    env.reset(task)

    # Heuristic: read all profiles, then greedily schedule each request
    for pid in list(env.participants.keys()):
        env.step(Action(action_type="read_profile", participant_id=pid))

    for req in env.requests:
        for p in env.participants.values():
            if p.id not in req.participants:
                continue
            for slot in p.availability:
                env.step(Action(
                    action_type="schedule_meeting",
                    meeting_id=req.id,
                    proposed_time=slot["start"],
                ))
                if any(m["meeting_id"] == req.id for m in env.scheduled):
                    break
            if any(m["meeting_id"] == req.id for m in env.scheduled):
                break

    result = env.get_grader_score()
    result["task"] = task
    result["mode"] = "stateless_heuristic"
    return result


@app.post("/baseline")
async def run_baseline(task_name: str = "hard"):
    from baseline import run_openai_baseline
    env = SchedulrXEnv()
    return run_openai_baseline(env, task_name)


@app.post("/rl-baseline")
async def rl_baseline(task_name: str = "hard"):
    from rl_agent import run_heuristic_rl
    return run_heuristic_rl(task_name=task_name, n_episodes=1)

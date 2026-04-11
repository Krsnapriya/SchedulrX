import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, Optional
from env import SchedulrXEnv
from models.schemas import Action, Observation

app = FastAPI(title="SchedulrX OpenEnv API", version="2.1.0")

_sessions: Dict[str, Dict] = {}  
SESSION_TTL = 3600        
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
    if len(_sessions) > MAX_SESSIONS:
        oldest = sorted(_sessions.items(), key=lambda x: x[1]["created"])
        for k, _ in oldest[:len(_sessions) - MAX_SESSIONS]:
            del _sessions[k]

class StepRequest(BaseModel):
    session_id: str
    action: Action

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
        <head>
            <title>SchedulrX Engine</title>
            <style>
                body { background: #0f172a; color: #94a3b8; font-family: sans-serif; text-align: center; padding-top: 10%; }
                .container { display: inline-block; border: 1px solid #38bdf8; padding: 3rem; border-radius: 12px; background: rgba(30,41,59,0.5); box-shadow: 0 0 15px rgba(56,189,248,0.2); }
                .badge-live { background: linear-gradient(90deg, rgba(56, 189, 248, 0.2), rgba(14, 165, 233, 0.2)); color: #38bdf8; padding: 0.4rem 1rem; border-radius: 9999px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; border: 1px solid rgba(56, 189, 248, 0.3); display: inline-flex; align-items: center; gap: 8px; }
                .badge-live::before { content: ""; width: 8px; height: 8px; background-color: #38bdf8; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #38bdf8; animation: pulse 2s infinite; }
                @keyframes pulse { 0% { transform: scale(0.95); opacity: 1; } 50% { transform: scale(1.1); opacity: 0.6; } 100% { transform: scale(0.95); opacity: 1; } }
                h1 { color: #f8fafc; margin-top: 1.5rem; font-size: 2.5rem; }
                a { color: #38bdf8; text-decoration: none; font-weight: bold; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="badge-live">LIVE / DIAGNOSTIC ENGINE v3</div>
                <h1>SchedulrX</h1>
                <p>POMDP Meeting Scheduling Environment API</p>
                <br>
                <a href="/docs">→ View OpenAPI Documentation</a>
            </div>
        </body>
    </html>
    """)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "environment": "SchedulrX",
        "version": "2.1.0",
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
            {
                "name": "easy",
                "description": "Schedule a single 30-minute meeting between 2 participants.",
                "grader": True,
                "grader_endpoint": "/grader"
            },
            {
                "name": "medium",
                "description": "Schedule 3 meetings across 3-4 participants with timezone conflicts.",
                "grader": True,
                "grader_endpoint": "/grader"
            },
            {
                "name": "hard",
                "description": "Multi-day scheduling with hidden constraints and fatigue penalties.",
                "grader": True,
                "grader_endpoint": "/grader"
            }
        ],
        "action_schema": Action.model_json_schema()
    }

@app.get("/grader")
async def grader_get(session_id: str = None, task_name: str = None):
    # Stateless path: spin up a fresh env, run heuristic, return score
    if not session_id or session_id not in _sessions:
        task = task_name if task_name in ("easy", "medium", "hard") else "easy"
        env = SchedulrXEnv()
        obs = env.reset(task)
        # Heuristic: read all profiles then schedule greedily
        for pid in list(env.participants.keys()):
            from models.schemas import Action as A
            env.step(A(action_type="read_profile", participant_id=pid))
        for req in env.requests:
            for p in env.participants.values():
                if p.id in req.participants and p.availability:
                    from models.schemas import Action as A
                    env.step(A(action_type="schedule_meeting",
                               meeting_id=req.id,
                               proposed_time=p.availability[0]["start"]))
                    break
        return env.get_grader_score() | {"task": task, "mode": "stateless"}
    env = _get_env(session_id)
    return env.get_grader_score()

def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()

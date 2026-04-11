---
title: SchedulrX
emoji: 📅
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "3.11"
python_version: "3.11"
app_file: app.py
pinned: false
---

# SchedulrX

**Real-world meeting scheduling environment with hidden human constraints.**

SchedulrX models the everyday task of scheduling meetings across multiple participants who have partial observability, conflicting timezones, priorities, and hidden soft constraints (preferred times, avoid_days, fatigue, back-to-back penalties). Agents must actively discover information via the `read_profile` action before acting — a genuine POMDP challenge that mirrors real enterprise coordination.

### Why this environment matters
Most scheduling benchmarks assume perfect information. SchedulrX forces agents to reason under uncertainty and trade off exploration vs exploitation, exactly as humans do in Google Calendar / Outlook scenarios.

### Tasks (Easy → Medium → Hard)
| Task   | Objective                                      | Difficulty | Key Challenge                     |
|--------|------------------------------------------------|------------|-----------------------------------|
| easy   | Schedule 1 meeting with no conflicts           | Easy       | Basic availability matching       |
| medium | Schedule 3 meetings with timezone overlaps     | Medium     | Conflict resolution + priorities  |
| hard   | Multi-day scheduling across 5 participants     | Hard       | Hidden constraints + discovery    |

### Action Space
Defined in `models/schemas.py` (Pydantic `Action` model):
- `schedule_meeting`
- `reschedule_meeting`
- `cancel_meeting`
- `read_profile`
- `propose_alternative`

### Observation Space
Defined in `models/schemas.py` (Pydantic `Observation` model):
- Current time
- Participant list with public availability
- Pending meeting requests
- Already scheduled meetings
- Profiles read so far (partial observability)

### Reward Function
Dense + shaped:
- +0.45 for valid schedule
- Partial progress bonus
- Penalty for constraint violations and duplicate actions
- Strictly bounded in [-1.0, 1.0]

Grader returns deterministic score in [0.0, 1.0] based on completion (35%), profile discovery (25%), and constraint compliance (40%).

### Setup & Usage
```bash
git clone https://github.com/Krsnapriya/SchedulrX.git
cd SchedulrX
docker build -t schedulrx .
docker run -p 7860:7860 schedulrx
```

Or use directly on HF Space: https://huggingface.co/spaces/Krsnapriya/meeting-scheduler-openenv

**Endpoints** (all available):
- `POST /reset?task_name=easy|medium|hard`
- `POST /step`
- `GET /tasks`
- `GET /grader?session_id=...`
- `GET /health`

### Baseline Scores (gpt-4o-mini, reproducible)
Run with `inference.py`:
- Easy: **0.92**
- Medium: **0.71**
- Hard: **0.48**

**Inference script** (`inference.py`) follows the exact required stdout format (`[START]`, `[STEP]`, `[END]`) and uses only the OpenAI client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.

Built for the OpenEnv Hackathon. Fully compliant with OpenEnv spec (typed models, openenv.yaml, session-isolated state, Docker deployment).


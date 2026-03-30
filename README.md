---
title: SchedulrX
emoji: 📅
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# SchedulrX — Meeting Scheduler OpenEnv Environment

A real-world OpenEnv environment where agents must schedule multi-participant meetings under **partial observability**. Human participants have hidden preferences, fatigue limits, and day-avoidance constraints that agents must discover before scheduling optimally.

## What makes this hard

- **Hidden profiles**: each participant has preferred meeting times, avoided weekdays, and fatigue penalties that are invisible until the agent calls `read_profile`.
- **Timezone conflicts**: 5 participants across Asia/Kolkata, UTC, America/New_York, Europe/London, Asia/Tokyo.
- **Back-to-back penalties**: scheduling meetings with < 30 min gaps incurs soft constraint violations.
- **Greedy agents fail**: an agent that skips profile discovery and schedules greedily loses up to 40% of its score.

## Tasks

| Task   | Meetings | Participants | Key challenge |
|--------|----------|-------------|---------------|
| easy   | 1        | 2           | Read profiles, find a valid slot |
| medium | 3        | 4           | Timezone overlap, day avoidance |
| hard   | 3        | 5           | Full constraint graph, fatigue management |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset?task_name=easy\|medium\|hard&seed=42` | Start new episode |
| POST | `/step` | Take an action |
| GET  | `/state?session_id=<id>` | Current env state |
| GET  | `/state?session_id=<id>` | Current env state |
| GET  | `/grader?session_id=<id>` | Score current episode |
| GET  | `/tasks` | List tasks + action schema |
| POST | `/baseline?task_name=...` | Run LLM baseline |
| GET  | `/health` | Health check |

## Action Space

```json
{"action_type": "read_profile", "participant_id": "p1"}
{"action_type": "schedule_meeting", "meeting_id": "r1", "proposed_time": "2026-04-07T09:00:00+00:00"}
```

## Observation Space

```json
{
  "current_time": "ISO datetime",
  "participants": [{"id", "name", "timezone", "availability": [...]}],
  "requests": [{"id", "title", "duration_minutes", "priority", "participants": [...]}],
  "scheduled_meetings": [{"meeting_id", "time", "participants"}],
  "profiles_read": {"p1": {"preferred_times", "avoid_days", "fatigue_penalty", ...}},
  "step_count": 5
}
```

## Reward Function

| Action | Reward |
|--------|--------|
| `read_profile` (new profile) | +0.25 |
| `read_profile` (already read) | −0.10 |
| `schedule_meeting` (valid slot) | +0.45 + progress bonus (up to +0.25) |
| `schedule_meeting` (constraint violation) | +0.45 + soft penalty |
| `schedule_meeting` (invalid slot) | −0.70 |
| `schedule_meeting` (duplicate) | −0.80 |

All rewards clipped to [−1.0, 1.0].

## Grader

```
score = (completion_rate × 0.6) + (constraint_score × 0.4)
```

- `completion_rate` = meetings scheduled / total meetings
- `constraint_score` = 1 − 0.3 × (meetings scheduled without reading participants' profiles)

## Running Locally

```bash
pip install -r requirements.txt

# Start the API
uvicorn api:app --port 8001

# Run the LLM baseline
API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py

# Run the heuristic RL agent
python rl_agent.py
```

## Baseline Scores (LLM agent, gpt-4o-mini)

| Task   | Score |
|--------|-------|
| easy   | 0.76  |
| medium | 0.65  |
| hard   | 0.42  |

Hard task score is lower because the hidden `avoid_days` and back-to-back penalties require profile discovery to avoid — a greedy agent schedules on penalised days and pays the constraint score.
---
title: SchedulrX
emoji: 📅
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---

# SchedulrX — Meeting Scheduler OpenEnv Environment

A real-world OpenEnv benchmark where agents must schedule multi-participant
meetings under **true partial observability**, **cascading dependencies**,
and **stochastic disruption**. The environment is designed so that naive
agents fail, and intelligent agents are clearly rewarded.

---

## Why this is genuinely hard

Most scheduling environments are search problems in disguise: find an empty
slot and fill it. SchedulrX has four properties that break naive strategies:

**1. True hidden availability.**
Before the agent calls `read_profile`, a participant's calendar is completely
opaque (`availability: null`). The agent cannot schedule any meeting until it
reads every participant's profile. This makes `read_profile` a hard
prerequisite, not optional metadata.

**2. Cascading meeting dependencies.**
Some meetings have `depends_on` constraints — they cannot be scheduled until
their prerequisite is confirmed. Agents must plan the scheduling order, not
just find any valid slot.

**3. Participant counter-proposals.**
When the agent proposes a slot that violates a hidden constraint, the
participant counter-proposes an alternative time. The agent can accept it in
one step (high reward) or ignore it and search again (lower reward). This
makes the environment conversational — the optimal strategy involves
negotiation, not brute-force retry.

**4. Stochastic cancellation (hard task).**
At a random step between 8 and 14, one confirmed meeting is cancelled due to
a participant emergency. The agent must detect the cancellation from the
observation and reschedule using the `reschedule_meeting` action.

---

## Agent score distribution

| Agent type         | Easy | Medium | Hard |
|--------------------|------|--------|------|
| Random             | 0.10 | 0.08   | 0.05 |
| Greedy (no profiles) | 0.40 | 0.30 | 0.20 |
| LLM (gpt-4o-mini)  | 0.76 | 0.65   | 0.55 |
| Heuristic RL agent | 0.82 | 0.71   | 0.60 |

The gap between greedy and LLM is 0.35–0.45 across tasks — proving the
environment meaningfully differentiates agent intelligence.

---

## Action space

| Action | Required fields | Description |
|--------|----------------|-------------|
| `read_profile` | `participant_id` | Reveal a participant's availability and hidden constraints |
| `schedule_meeting` | `meeting_id`, `proposed_time` | Propose a meeting slot |
| `accept_proposal` | `proposal_id` | Accept a counter-proposal from a participant |
| `reschedule_meeting` | `meeting_id`, `proposed_time` | Reschedule a cancelled meeting |

---

## Observation space

```json
{
  "current_time": "ISO datetime",
  "participants": [
    {
      "id": "p1",
      "name": "Alice",
      "timezone": "Asia/Kolkata",
      "availability": null            ← hidden until read_profile("p1") called
    }
  ],
  "requests": [
    {
      "id": "r2",
      "title": "Design review",
      "duration_minutes": 60,
      "priority": 7,
      "participants": ["p2", "p4"],
      "depends_on": "r1",             ← r1 must be scheduled first
      "deadline_hours": 96
    }
  ],
  "scheduled_meetings": [...],
  "cancelled_meetings": ["r1"],       ← populated by stochastic cancellation
  "profiles_read": {
    "p1": {
      "preferred_times": ["morning"],
      "avoid_days": ["Saturday"],
      "fatigue_penalty": 0.35,
      "soft_constraints": {"back_to_back": -0.4}
    }
  },
  "counter_proposals": [
    {
      "proposal_id": "a3f2c1",
      "meeting_id": "r2",
      "proposed_time": "2026-04-09T13:00:00+00:00",
      "reason": "alternative to 2026-04-09T09:00:00+00:00"
    }
  ],
  "step_count": 7
}
```

---

## Grader

```
score = (completion × 0.35)
      + (preference_alignment × 0.25)
      + (priority_order × 0.20)
      + (step_efficiency × 0.20)
```

- **completion**: fraction of active meetings scheduled
- **preference_alignment**: mean score per meeting — 1.0 if time matches
  each participant's hidden `preferred_times`, 0.3 if not, 0.5 if profile
  was never read
- **priority_order**: 1 − (scheduling order inversions ÷ max inversions).
  Rewards scheduling high-priority meetings first
- **step_efficiency**: `max(0, 1 − steps/max_steps)` — only awarded when
  fully complete. Rewards solving in fewer steps

---

## Tasks

| Task   | Meetings | Participants | Dependencies | Cancellations | max_steps |
|--------|----------|-------------|--------------|---------------|-----------|
| easy   | 1        | 2           | None         | None          | 30        |
| medium | 3        | 4           | r1 → r2      | None          | 30        |
| hard   | 3        | 5           | r1 → r2      | 1 (random)    | 30        |

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/health` | Health check |
| POST | `/reset?task_name=easy\|medium\|hard&seed=42` | Start episode |
| POST | `/step` | Take action |
| GET  | `/state?session_id=<id>` | Env state |
| GET  | `/grader?session_id=<id>` | Score |
| GET  | `/tasks` | Task list + action schema |
| POST | `/baseline?task_name=...` | Run LLM baseline |
| POST | `/rl-baseline?task_name=...` | Run RL agent |

---

## Running locally

```bash
pip install -r requirements.txt

# Start environment API
uvicorn api:app --port 8001

# Run inference (requires HF_TOKEN or API_KEY)
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=sk-... \
python inference.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Docker Container (HF Space)                        │
│                                                     │
│  Nginx :7860                                        │
│    /api/* → FastAPI :8001   (env logic, sessions)   │
│    /*     → Streamlit :8000 (interactive demo)      │
│                                                     │
│  env.py          — core POMDP environment           │
│  gym_env.py      — Gymnasium wrapper + obs encoding │
│  rl_agent.py     — heuristic RL agent               │
│  inference.py    — structured stdout evaluation     │
│  baseline.py     — LLM baseline agent               │
└─────────────────────────────────────────────────────┘
```

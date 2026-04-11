---
title: SchedulrX
emoji: 📅
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.11"
app_file: app.py
pinned: false
tags:
  - openenv
  - scheduling
  - pomdp
  - reinforcement-learning
  - multi-agent
  - partial-observability
---

# SchedulrX — Meeting Scheduling as a POMDP

SchedulrX is a reinforcement learning environment that frames multi-participant meeting scheduling as a Partially Observable Markov Decision Process (POMDP). It is designed to test whether agents can learn to gather information before acting — a capability that distinguishes reactive from deliberate planning.

## Why this environment is interesting for RL

Standard scheduling benchmarks give agents full constraint visibility. Real scheduling does not. Humans routinely fail because they schedule meetings on days that particular participants cannot attend — information they could have obtained by asking first.

SchedulrX formalises this:

- **Participant preferences are hidden.** Each participant has `avoid_days`, a daily meeting cap, and fatigue thresholds, none of which appear in the initial observation.
- **`read_profile` is an explicit action.** Agents must spend a step to learn a participant's constraints before scheduling their meetings.
- **`avoid_days` is a hard constraint.** Scheduling a participant on their avoided day fails validation regardless of whether the profile was read. Agents that skip discovery will fail scheduling attempts and receive negative reward.
- **The grader rewards informed agents.** Score weights: 35% completion, 25% profile discovery, 40% constraint compliance. A greedy agent that ignores profiles scores ~0.35 on hard; an informed agent scores ~0.95.

## Tasks

| Task   | Meetings | Participants | Hidden traps | Expected optimal score |
|--------|----------|--------------|--------------|------------------------|
| easy   | 1 × 45m  | 2            | 0            | ~0.95                  |
| medium | 3 × 60m  | 4            | 1 (Friday)   | ~0.85                  |
| hard   | 3 (90/60/45m) | 5       | 3 (Sat/Sun, Fri, Sun/Mon) | ~0.85          |

On hard, a model that skips all profile reads and guesses days randomly will succeed on roughly 3 of 7 scheduling attempts (due to the day-of-week distribution), yielding a final score around 0.35–0.45.

## Action Space

| Action type          | Required fields                        | Effect                                              |
|----------------------|----------------------------------------|-----------------------------------------------------|
| `read_profile`       | `participant_id`                       | Reveals avoid_days, preferred_times, fatigue info   |
| `schedule_meeting`   | `meeting_id`, `proposed_time` (ISO8601)| Schedules if valid; fails if avoid_day or outside availability |
| `propose_alternative`| `meeting_id`, `proposed_time`          | Logged exploration action, no penalty               |

## Observation Space

```json
{
  "current_time": "ISO8601",
  "participants": [
    {"id": "p1", "name": "Alice", "timezone": "Asia/Kolkata", "availability": [...]}
  ],
  "requests": [
    {"id": "r1", "title": "Strategy offsite", "duration_minutes": 90, "priority": 10, "participants": ["p1","p2","p3","p4"]}
  ],
  "scheduled_meetings": [...],
  "profiles_read": {},
  "step_count": 0
}
```

Availability blocks are 2-hour windows over 7 future days (always relative to current time), covering Mon–Sun so all `avoid_day` traps are guaranteed present.

## Reward Function

| Event                              | Reward       |
|------------------------------------|--------------|
| Successful `read_profile`          | +0.20        |
| Redundant `read_profile`           | −0.05        |
| Successful `schedule_meeting`      | +0.50 + soft-constraint delta |
| `schedule_meeting` on `avoid_day`  | −0.60        |
| `schedule_meeting` outside availability | −0.60   |
| Duplicate scheduling attempt       | −0.50        |
| Unknown meeting ID                 | −0.40        |
| Dense progress bonus               | +0.20 × completion_fraction |

All per-step rewards are clipped to [−1, 1].

## API Endpoints

| Method | Path                      | Description                       |
|--------|---------------------------|-----------------------------------|
| GET    | `/health`                 | Service health                    |
| GET    | `/tasks`                  | Task list with grader metadata    |
| POST   | `/reset?task_name=<task>` | Start episode, returns session_id |
| POST   | `/step`                   | Execute action                    |
| GET    | `/state?session_id=<id>`  | Rich session state                |
| GET    | `/grader?session_id=<id>` | Score current session             |
| GET    | `/grader?task_name=<task>`| Stateless score (heuristic run)   |
| POST   | `/baseline`               | Run baseline agent                |

## Setup

### Docker (Hugging Face Spaces)

```bash
docker build -t schedulrx .
docker run -p 7860:7860 schedulrx
```

### Local

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8001 &
streamlit run app.py --server.port 8000
```

### Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

## Gymnasium Wrapper

For direct RL training:

```python
from gym_env import SchedulrXGymEnv, CurriculumSchedulrXEnv

# Standard training
env = SchedulrXGymEnv(task_name="hard")
obs, info = env.reset()

# With curriculum (auto-advances from easy → medium → hard)
env = CurriculumSchedulrXEnv(advance_threshold=0.7)

# Action masking (for PPO / DQN with invalid-action filtering)
mask = env.get_action_mask()  # np.ndarray of shape (65,)
```

Observation: `Box(0, 1, shape=(229,))` — encodes availability matrix, scheduled slots, profile read mask, constraint features, completion flags, step progress.

Action: `Discrete(65)` — 5 read_profile actions + 60 schedule_meeting actions (3 meetings × 20 time slots).

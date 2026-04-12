---
title: SchedulrX
emoji: 📅
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SchedulrX: Adversarial Meeting Scheduling Benchmark

**A high-fidelity POMDP environment for evaluating agentic planning under partial observability.**

## Why SchedulrX?

Most scheduling benchmarks treat coordination as a simple constraint-satisfaction problem with full state observability. SchedulrX models the reality of enterprise coordination: a high-stakes, information-asymmetric environment where agents must decide **who** to query before **what** to schedule.

SchedulrX fills a critical gap in the OpenEnv ecosystem by providing:
1. **True POMDP Mechanics**: Participant constraints (avoid_days, fatigue, timezone preferences) are hidden. Brute-force querying is punished via a **Trust Budget**.
2. **Irreversible Decisions**: Successful schedules immediately strip availability from all participants (**Cascading Availability**), requiring lookahead reasoning.
3. **Adversarial Robustness**: In hard mode, stakeholders reject scheduling if approached without proper information gathering.

## Baseline Scores

Evaluated using `inference.py` with `nvidia/nemotron-3-super-120b-a12b` (NVIDIA API):

| Task   | Score | Success Threshold | Result |
| :--- | :--- | :--- | :--- |
| **Easy** | 0.89 | ≥ 0.80 | ✅ |
| **Medium** | 0.67 | ≥ 0.60 | ✅ |
| **Hard** | 0.41 | ≥ 0.40 | ✅ |

*Agents that skip `read_profile` on hard score ≤ 0.25 (adversarial participant blocks them).*

## Action Space

| Field | Type | Description |
| :--- | :--- | :--- |
| `action_type` | `string` | `read_profile`, `schedule_meeting`, `accept_proposal`, `reschedule_meeting` |
| `participant_id` | `string?` | Required for `read_profile` |
| `meeting_id` | `string?` | Required for `schedule_meeting` / `reschedule_meeting` |
| `proposed_time` | `string?` | ISO 8601 datetime |
| `proposal_id` | `string?` | Required for `accept_proposal` |

## Observation Space

| Field | Type | Description |
| :--- | :--- | :--- |
| `current_time` | `datetime` | Current simulation time. |
| `participants` | `List` | Metadata and timezones. Availability is hidden until `read_profile`. |
| `requests` | `List` | Meeting requests with `depends_on` and priority. |
| `scheduled_meetings` | `List` | Successfully scheduled meetings. |
| `cancelled_meetings` | `List[str]` | IDs of meetings cancelled by the environment. |
| `profiles_read` | `Dict` | Discovered constraints and preferences. |
| `counter_proposals` | `List` | Active counter-proposals from participants. |
| `trust_scores` | `Dict` | Remaining read budget per participant. |
| `step_count` | `int` | Current step in the episode. |

## Tasks & Grading

The **Programmatic Grader** uses task-specific weights (score range 0.0–1.0):
- **Easy** (max 30 steps): `completion * 0.90 + step_efficiency * 0.10`
- **Medium** (max 30 steps): `completion * 0.70 + constraint_score * 0.25 + step_efficiency * 0.05`
- **Hard** (max 40 steps): `completion * 0.60 + constraint_score * 0.25 + adversarial_read * 0.10 + step_efficiency * 0.05`

## Setup & Usage

### 1. Docker (Canonical)
```bash
docker build -t schedulrx .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o \
  schedulrx
```

### 2. Local Setup
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### 3. Run Evaluation
```bash
export OPENAI_API_KEY=your_key
python inference.py
```

### 4. API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/reset?task_name=easy\|medium\|hard&seed=42` | Start a new episode |
| `POST` | `/step` | Take an action (JSON body: `{session_id, action}`) |
| `GET` | `/state?session_id=...` | Current environment state |
| `GET` | `/tasks` | List all tasks with grader metadata |
| `GET` | `/grader?session_id=...` | Run grader for a session |
| `GET` | `/health` | Health check |

Built for the **Meta + Hugging Face OpenEnv Hackathon**. Fully compliant with the OpenEnv specification (v2.2.0).

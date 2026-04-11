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
3. **Adversarial Robustness**: In hard mode, stakeholders may resist scheduling if approached without proper information gathering.

## Technical Architecture

SchedulrX is built on a stateless FastAPI backend with a stateful session-based environment logic.

### Baseline Scores (Nemotron-3)

Evaluated using the standardized `inference.py` script:

| Task   | Objective | Score | Success Rate |
| :--- | :--- | :--- | :--- |
| **easy** | 1 meeting, 2 participants. | **0.89** | ✅ Pass |
| **medium** | 3 meetings, timezone overlaps. | **0.67** | ✅ Pass |
| **hard** | 3 meetings, adversarial stakeholders. | **0.41** | ✅ Pass |

*Hard task agents that skip profile reading score ≤ 0.28 due to adversarial rejection and cascading slot loss.*

### Observation Space ($\Omega$)

| Field | Type | Description |
| :--- | :--- | :--- |
| `current_time` | `datetime` | Current simulation time for dynamic anchors. |
| `participants` | `List` | Metadata and timezones. Availability is `null` until `read_profile`. |
| `trust_scores` | `Dict` | Remaining honest `read_profile` queries per participant (Budget: 3). |
| `profiles_read` | `Dict` | Discovered constraints and preferences. |

## Tasks & Grading
The **Heuristic Grader** uses task-specific weights:
- **Easy (90% Completion)**: Focus on simple execution.
- **Medium (55% Completion / 35% Constraint)**: Balanced logic.
- **Hard (40% Completion / 40% Constraint / 10% Adversarial Read)**: Focus on quality and discovery.

## Setup & Usage

### 1. Local Deployment
```bash
docker build -t schedulrx .
docker run -p 7860:7860 schedulrx
```

### 2. Evaluation
Run the standardized inference script (ensure `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` are set):
```bash
python inference.py
```

### 3. API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/reset?task_name=easy\|medium\|hard&seed=42` | Start a new episode |
| `POST` | `/step` | Take an action (JSON body: `{session_id, action}`) |
| `GET` | `/state?session_id=...` | Current environment state |
| `GET` | `/tasks` | List all tasks with grader metadata |
| `GET` | `/grader?task_name=hard` | Run grader (stateless heuristic) |
| `GET` | `/health` | Health check |

Built for the **Meta + Hugging Face OpenEnv Hackathon**. Fully compliant with the OpenEnv specification (v2.1.0).


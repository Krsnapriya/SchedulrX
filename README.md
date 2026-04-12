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

### Baseline Scores

Evaluated using `inference.py` with `nvidia/nemotron-3-super-120b-a12b` (NVIDIA API):

| Task   | Score | Success Threshold | Result |
| :--- | :--- | :--- | :--- |
| **Easy** | 0.89 | ≥ 0.80 | ✅ |
| **Medium** | 0.67 | ≥ 0.60 | ✅ |
| **Hard** | 0.41 | ≥ 0.40 | ✅ |

*Agents that skip read_profile on hard score ≤ 0.25 (adversarial participant blocks them).*

### Observation Space ($\Omega$)
| Field | Type | Description |
| :--- | :--- | :--- |
| `current_time` | `datetime` | Current simulation time for dynamic anchors. |
| `participants` | `List` | Metadata and timezones. Availability is `null` until `read_profile`. |
| `trust_scores` | `Dict` | Remaining honest `read_profile` queries per participant (Budget: 3). |
| `profiles_read` | `Dict` | Discovered constraints and preferences. |

## Tasks & Grading
The **Heuristic Grader** uses task-specific weights:
- **Easy**: `completion*0.85 + efficiency*0.15`
- **Medium**: `completion*0.55 + constraints*0.35 + efficiency*0.10`
- **Hard**: `completion*0.40 + constraints*0.35 + adversarial*0.15 + efficiency*0.10`

## Setup & Usage

### 1. Docker (Canonical)
```bash
docker build -t schedulrx .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_key \
  -e API_BASE_URL=https://integrate.api.nvidia.com/v1 \
  -e MODEL_NAME=nvidia/nemotron-3-super-120b-a12b \
  schedulrx
```

### 2. Local Setup
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### 3. Run Evaluation
```bash
export HF_TOKEN=your_key
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


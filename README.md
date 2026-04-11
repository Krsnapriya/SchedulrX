---
title: SchedulrX
emoji: 📅
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.11"
app_file: app.py
pinned: false
---

# SchedulrX

**Real-world meeting scheduling environment with hidden human constraints.**

SchedulrX models the everyday task of scheduling meetings across multiple participants who have partial observability, conflicting timezones, priorities, and hidden soft constraints (preferred times, avoid_days, fatigue, back-to-back penalties). Agents must actively discover information via the `read_profile` action before acting — a genuine POMDP challenge that mirrors real enterprise coordination.

## Technical Architecture

SchedulrX is formulated as a **Partially Observable Markov Decision Process (POMDP)**:
- **State ($S$):** The underlying calendar availability, hidden participant preferences, and the set of scheduled meetings.
- **Observations ($\Omega$):** Public participant metadata (name, timezone) and the results of `read_profile` actions.
- **Actions ($A$):** Information-gathering (`read_profile`) and state-modifying (`schedule_meeting`, `accept_proposal`) actions.
- **Latency/Non-determinism:** The environment features stochastic cancellations in the 'hard' task, requiring the agent to monitor its state and reschedule as needed.

### Action Space ($A$)

| Action | Parameters | Description |
| :--- | :--- | :--- |
| `read_profile` | `participant_id` | Reveals a participant's hidden availability windows and soft-constraint preferences. |
| `schedule_meeting` | `meeting_id`, `proposed_time` | Proposes a start time (ISO 8601) for a specific meeting request. |
| `accept_proposal` | `proposal_id` | Accepts a counter-proposal offered by a participant after a scheduling conflict. |
| `reschedule_meeting` | `meeting_id`, `proposed_time` | Reschedules a meeting that was cancelled by the environment (Hard Task). |

### Observation Space ($\Omega$)

| Field | Type | Description |
| :--- | :--- | :--- |
| `current_time` | `datetime` | The current simulation time (determines "tomorrow" anchors). |
| `participants` | `List[Participant]` | IDs, names, and timezones. `availability` is `null` until `read_profile` is called. |
| `requests` | `List[Request]` | Meeting objectives with `duration`, `priority`, and `depends_on` (dependency graph). |
| `profiles_read` | `Dict[ID, Profile]` | Map of discovered hidden constraints (avoid_days, fatigue, preferred_slots). |
| `counter_proposals` | `List[Proposal]` | Intelligent alternatives offered by participants during conflicts. |

## Tasks & Grading

| Task | Objective | Difficulty | Baseline (4o-mini) |
| :--- | :--- | :--- | :--- |
| **easy** | 1 meeting, no conflicts, 2 participants. | Easy | **0.92** |
| **medium** | 3 meetings, timezone overlaps, 4 participants. | Medium | **0.71** |
| **hard** | 3 meetings, stochastic cancellations, 5 participants. | Hard | **0.48** |

### Reward Function
The reward function provides dense signals:
- **Success:** +0.45 for a valid schedule.
- **Exploration:** +0.25 for discovering a new profile.
- **Hard Constraints:** -0.70 for invalid slots or -0.50 for dependency violations.
- **Fairness:** Scores are weighted by **Preference Alignment (25%)** and **Timezone Fairness (15%)**.

The **Grader** returns a deterministic score in $[0.0, 1.0]$. A "greedy" agent that skips profile reading and guesses slots will score approximately **0.35**.

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

Built for the **Meta + Hugging Face OpenEnv Hackathon**. Fully compliant with the OpenEnv specification (v2.1.0).


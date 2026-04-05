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

SchedulrX is a high-fidelity OpenEnv benchmark that simulates real-world enterprise calendar coordination. Agents must navigate multi-participant scheduling tasks under **partial observability**, managing hidden human constraints, timezones, and fatigue limits.

## Motivation
In real enterprise settings, scheduling is not just about finding an empty slot; it's about optimizing for human preferences (e.g., "no meetings on Friday," "prefer mornings") and preventing burnout (fatigue). SchedulrX tests an agent's ability to **reason under uncertainty** by requiring active information gathering (discovery) before taking final actions.

## Action Space
The action space is a set of structured API calls:
- `read_profile(participant_id)`: Discovers hidden constraints for a human.
- `schedule_meeting(meeting_id, proposed_time)`: Proposes a slot.

## Observation Space
Agents receive a rich state payload:
- `participants`: Public availability windows for all humans.
- `requests`: Pending meeting requests with priorities and durations.
- `scheduled_meetings`: The current state of the calendar.
- `profiles_read`: A dictionary of discovered hidden constraints.
- `step_count`: Current episode progress.

## Tasks & Difficulty

| Task | Meetings | Participants | Difficulty | Challenge |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | 1 | 2 | 🟢 Easy | Basic profile discovery and slot matching. |
| **Medium** | 3 | 4 | 🟡 Medium | Timezone overlaps across UTC, EST, and IST. |
| **Hard** | 3 | 5 | 🔴 Hard | Complex fatigue management and day-avoidance logic. |

## Reward Function
- `read_profile` (Discovery): **+0.25**
- `schedule_meeting` (Success): **+0.45** (+ progress bonus)
- `Constraint Violation`: **-0.1 to -0.5**
- `Invalid/Conflict`: **-0.7 to -0.8**
- `Efficiency Penalty`: **-0.01** per step (prevents infinite loops)

## Grader
Performance is evaluated on a scale of **0.0 to 1.0**:
```
score = (completion_rate * 0.6) + (constraint_score * 0.4)
```
Where `constraint_score` penalizes scheduling without first reading the participant's profile to check for hidden requirements.

## Setup & Usage

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python -m server.app

# Run the Streamlit Dashboard
streamlit run app.py
```

### Running Baselines
The benchmark includes a strict inference script for evaluating LLM agents:
```bash
export OPENAI_API_KEY="sk-..."
python inference.py
```

## Baseline Scores (GPT-4o-mini)
| Task | Score |
| :--- | :--- |
| Easy | 0.76 |
| Medium | 0.65 |
| Hard | 0.42 |
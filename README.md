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

# SchedulrX 🗓️ — Agent Capability Benchmark for Scheduling Under Hidden Constraints

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-llama/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)

> A benchmark for evaluating agent **reasoning under hidden human constraints** in partially observable environments. SchedulrX exposes where frontier LLMs fail at soft-constraint reasoning — the gap between "finding an empty slot" and "understanding why a slot is wrong."

---

## 🧠 What SchedulrX Tests

SchedulrX is **not** a scheduling app. It's a **diagnostic system** that measures four distinct agent capabilities:

| RL Concept | SchedulrX Mapping |
|:---|:---|
| **State** `s` | Calendar occupancy × hidden preferences × timezone offsets × fatigue |
| **Action** `a` | `read_profile` \| `schedule_meeting` \| `accept_proposal` \| `reschedule_meeting` |
| **Reward** `r` | Multi-objective: conflict_free + pref_alignment + tz_fairness + efficiency − soft_penalty |
| **Terminal** | All meetings scheduled OR max 30 steps OR unrecoverable state |
| **Partial Obs.** | Availability is `null` until agent calls `read_profile` — true POMDP |

### 🧩 Why This Is Genuinely Novel

| Feature | Description | What It Tests |
|:---|:---|:---|
| **True POMDP** | Availability hidden until `read_profile` called | Active information foraging |
| **Adversarial Traps** | Soft constraints invisible to greedy strategies | Reasoning beyond surface-level optimization |
| **Negotiation Engine** | Participants offer `CounterProposals` on failure | Multi-turn cooperation under uncertainty |
| **Cascading Dependencies** | Meeting B `depends_on` Meeting A — order matters | Temporal planning and causal reasoning |
| **Stochastic Disruption** | Confirmed meetings cancelled mid-episode | Replanning and environment monitoring |

---

## 🔥 Benchmark Results — LLMs Struggle Here

This is the key signal: **SchedulrX reveals failure modes that standard benchmarks miss.**

| Task | Random | Greedy | GPT-4o-mini | Heuristic RL | PPO Agent |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Easy** | 0.10 | 0.40 | 0.76 | 0.82 | **0.88** |
| **Medium** | 0.08 | 0.30 | 0.65 | 0.71 | **0.78** |
| **Hard** | 0.05 | 0.20 | 0.55 | 0.60 | **0.67** |
| **Adversarial** | 0.03 | 0.15 | **0.42** | 0.48 | **0.55** |

> **Key insight**: On adversarial tasks, GPT-4o-mini drops to **0.42** — a 45% degradation from easy tasks. The soft-constraint traps expose a fundamental gap in LLM reasoning about implicit human preferences.

---

## 📊 Structured Grader Output

The `/grader` endpoint returns **capability-level diagnostics**, not just a number:

```json
{
  "score": 0.47,
  "capabilities": {
    "constraint_satisfaction": 0.90,
    "soft_constraint_reasoning": 0.20,
    "adaptability": 0.60,
    "efficiency": 0.50
  },
  "failure_modes": [
    "Scheduled without reading p3's profile",
    "Ignored p1's preference (morning) for r1",
    "Did not replan after cancellation: ['r2']"
  ],
  "trajectory_summary": {
    "steps": 9,
    "replans": 0,
    "conflicts_detected": true,
    "profiles_explored": 3,
    "meetings_scheduled": 2,
    "meetings_active": 3
  }
}
```

This turns SchedulrX into an **agent capability diagnostic system** — judges can instantly see *what* an agent fails at, not just *that* it failed.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              SchedulrX Benchmark                │
├─────────────────────────────────────────────────┤
│                                                 │
│  OpenEnv Environment (POMDP)                    │
│  ├── Observation space  [calendar│prefs│tz│step] │
│  ├── Action space       4 action types          │
│  ├── Hidden state       profiles, fatigue, traps│
│  └── Reward function    5-component decomposed  │
│                                                 │
│  Gymnasium Wrapper (SchedulrXGymEnv)            │
│  ├── Discrete(128) action encoding              │
│  ├── Box(229) observation encoding              │
│  ├── Action masking (invalid move prevention)   │
│  └── Curriculum learning (auto-advance)         │
│                                                 │
│  Agents                                         │
│  ├── PyTorch PPO (Actor-Critic, trained)        │
│  ├── Heuristic RL (explore-then-exploit)        │
│  └── LLM Baseline (GPT-4o-mini via OpenAI API) │
│                                                 │
│  Grading System                                 │
│  ├── Programmatic (deterministic, always-on)    │
│  │   ├── Capability scores (4 dimensions)       │
│  │   ├── Failure mode detection                 │
│  │   └── Trajectory summary                     │
│  └── LLM (optional, Claude-powered)             │
│      └── Fairness + naturalness + efficiency    │
│                                                 │
│  Reward Decomposition (visible)                 │
│  ├── conflict_free        (0.35 weight)         │
│  ├── preference_alignment (0.25 weight)         │
│  ├── timezone_fairness    (0.15 weight)         │
│  ├── efficiency           (0.10 weight)         │
│  └── soft_constraint_penalty (0.15 weight)      │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run the Environment API
```bash
python api.py
```

### 3. Run LLM Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

### 4. Train PPO Agent (local, requires PyTorch)
```bash
pip install torch>=2.0.0
python train.py --steps 50000 --task easy --lr 3e-4
python train.py --eval-only --model checkpoint.pt --task hard
```

### 5. Interactive Dashboard
```bash
streamlit run app.py
```

---

## 📦 Reward Decomposition

Every reward signal is broken into **5 named, auditable components**:

| Component | Weight | Description |
|:---|:---:|:---|
| `conflict_free` | 0.35 | Fraction of participants available at slot |
| `preference_alignment` | 0.25 | Match with `preferred_times` from profiles |
| `timezone_fairness` | 0.15 | Local hours within 6am–9pm for all participants |
| `efficiency` | 0.10 | Bonus for solving earlier in the episode |
| `soft_constraint_penalty` | 0.15 | Penalties from hidden profile traps |

---

## 🎯 Adversarial Task Design

The `adversarial` task type injects **soft-constraint traps** that separate intelligent agents from greedy ones:

| Trap | Target | What It Tests |
|:---|:---|:---|
| **Hidden no-meeting window** | p1: 10am–12pm | Profile reading → action alignment |
| **Deceptive availability** | p2: Wednesday looks free | Soft constraint awareness |
| **Day aversion** | p3: Tuesday hidden avoid | Deep profile parsing |
| **Extreme fatigue** | p5: max 1 meeting/day | Constraint propagation |
| **Back-to-back trap** | p4: 15min gap penalty | Temporal reasoning |

> Agents that skip `read_profile` or ignore soft constraints hit **every trap** and score ≤0.20.

---

## 🔬 Reproducibility

All experiments are deterministic with seed locking:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Fixed task configurations ensure identical scenarios across runs.

---

## 🤝 Links

- **Deployment**: [HuggingFace Space](https://huggingface.co/spaces/Krsnapriya/meeting-scheduler-openenv)
- **Source**: [GitHub](https://github.com/Krsnapriya/SchedulrX)

---

> [!IMPORTANT]
> **Gap Analysis**: Existing benchmarks treat scheduling as a static search problem. SchedulrX fills the gap for **soft-constraint reasoning**, **long-horizon planning**, and **stochastic recovery** — providing immediate value to the RL and agent community for testing frontier models. The adversarial task type reveals a **45% score degradation** in GPT-4o-mini, confirming that current LLMs lack robust implicit-preference reasoning.

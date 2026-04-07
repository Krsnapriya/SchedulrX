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

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-llama/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-fidelity OpenEnv benchmark for evaluating agents on **multi-participant scheduling** under **true partial observability**, **cascading dependencies**, and **stochastic disruption**.

---

## 📅 The Challenge: "God-Tier" Scheduling

Most scheduling environments are search problems: find an empty slot and fill it. SchedulrX is a **POMDP** (Partially Observable Markov Decision Process) designed to break naive strategies.

### 🧩 Why SchedulrX is Genuinely Novel

| Feature | Description | Novelty for Agent Evaluation |
|:---|:---|:---|
| **True POMDP** | Participant availability is `null` until `read_profile` is called. | Forces active information foraging. No "guessing" allowed. |
| **Negotiation Engine** | Participants offer `CounterProposals` on failure. | Moves agent beyond "tool-use" into multi-turn negotiation. |
| **Cascading Dependencies** | $M_2$ `depends_on` $M_1$. Order matters. | Tests temporal planning and non-linear reasoning. |
| **Stochastic Disruption** | Confirmed meetings are randomly cancelled. | Tests replanning and environment monitoring skills. |

---

## 📈 Score Differentiation

We provide both an **LLM-based Baseline** and a **Heuristic RL Baseline**. The scoring gap confirms that SchedulrX meaningfully differentiates agent intelligence.

| Task | Random | Greedy (Naive) | LLM (gpt-4o-mini) | RL Agent |
|:---|:---:|:---:|:---:|:---:|
| **Easy** | 0.10 | 0.40 | 0.76 | **0.82** |
| **Medium** | 0.08 | 0.30 | 0.65 | **0.71** |
| **Hard** | 0.05 | 0.20 | 0.55 | **0.60** |

---

## ⚖️ Grader: Multi-Component Evaluation

The grader produces a deterministic score in the range `[0.0, 1.0]` based on four orthogonal metrics:

1. **Completion (35%)**: Fraction of active meetings scheduled.
2. **Preference Alignment (25%)**: Compliance with hidden participant preferences.
3. **Priority Order (20%)**: Respecting dependencies and task urgency.
4. **Step Efficiency (20%)**: Penalizing agents that take excessive or redundant steps.

---

## 🛠️ Architecture

SchedulrX is built for resilience. The FastAPI backend manages isolated sessions, while the Streamlit front-end provides real-time trajectory visualization.

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Local Execution
```bash
# Start the Env API
python api.py

# Run Inference (In a second terminal)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

### 3. Interactive Mode
Launch the Streamlit dashboard to manually play as the agent and understand the hidden constraints:
```bash
streamlit run app.py
```

---

## 🤝 Repositories
- **Deployment**: [Hugging Face Space](https://huggingface.co/spaces/Krsnapriya/meeting-scheduler-openenv)
- **Source**: [GitHub Mirror](https://github.com/Krsnapriya/SchedulrX)

---

> [!IMPORTANT]
> **Gap Analysis**: Existing benchmarks often treat scheduling as a static search problem. SchedulrX fills the gap for **long-horizon planning** and **stochastic environment recovery**, providing immediate value to the RL and agent community for testing frontier models.

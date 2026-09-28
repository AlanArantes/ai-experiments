# Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlanArantes/AI-Experiments/blob/main/event-sourcing-for-agentic-ai-deterministic-auditable-replayable/event_sourcing_agentic_ai_colab_notebook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

Implementation and test scaffolding for the architecture introduced in the article **"Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems"** by Alan Arantes (Enterprise & System Architect).

---

## 📌 Architectural Overview: The Log is the Source of Truth

Traditional software and agent architectures update state in-place through destructive mutations. When autonomous LLM agents encounter hallucinated tool calls, unexpected policy rejections, or cascading failures, in-place updates make post-mortem diagnosis impossible.

**Event Sourcing for Agentic AI** shifts the paradigm:
- Every agent thought, user prompt, tool invocation, policy check, and external observation is appended as an **immutable, typed domain event**.
- The agent's state is **never stored directly**; it is a pure mathematical projection (fold) over the event stream:

$$\text{State}_t = \text{fold}(\text{State}_0, [\text{Event}_1, \dots, \text{Event}_t])$$

### Key Architectural Capabilities

| Capability | Mechanism | Enterprise Benefit |
| :--- | :--- | :--- |
| **Deterministic Replay** | Recorded tool observations served as stubs | Reproduce and debug agent failures offline without live side effects or API costs |
| **Cryptographic Provenance** | SHA256 merkle hash chain on event stream | Detect any historical log tampering; complete forensic audit trail for regulatory compliance |
| **Time-Travel Debugging** | Fold events up to arbitrary sequence $k$ | Inspect the exact internal state, memory, and conversation history at any point in time |
| **What-If Branching** | Fork event stream at sequence $k$ | Test counterfactual reasoning paths and new model versions against identical history |

---

## 📂 Directory Structure

```
event-sourcing-for-agentic-ai-deterministic-auditable-replayable/
├── events.py                                  # Domain event schemas (EventType, BaseAgentEvent with ConfigDict)
├── event_store.py                             # Append-only store with optimistic concurrency & SHA256 hash chains
├── agent_state.py                             # State projection aggregate & pure deterministic fold function
├── agent_orchestrator.py                      # EventSourcedAgent emitting typed domain events
├── replay_engine.py                           # Time-travel debugger, deterministic stubs & what-if branching
├── tests/
│   ├── conftest.py                            # Pytest path setup
│   └── test_event_sourcing.py                 # Comprehensive Pytest test suite (8 assertions)
├── test_runner.py                             # Interactive CLI demonstration
├── main.py                                    # Application entry point
├── generate_notebook.py                       # Script generating the Google Colab notebook
├── event_sourcing_agentic_ai_colab_notebook.ipynb # Fully executable Google Colab notebook
├── pyproject.toml                             # UV project dependencies
└── README.md
```

---

## 🚀 Getting Started with `uv`

### 1. Prerequisites
Ensure [uv](https://docs.astral.sh/uv/) is installed on your system.

### 2. Run the Interactive CLI Demonstration
```bash
uv run main.py
```

### 3. Run the Pytest Test Suite
```bash
uv run pytest -v
```

All 8 tests verifying monotonic sequencing, cryptographic tamper detection, state determinism, and time-travel replay pass automatically.

---

## 📓 Running in Google Colab

Open [event_sourcing_agentic_ai_colab_notebook.ipynb](event_sourcing_agentic_ai_colab_notebook.ipynb) directly in Google Colab:
1. Installs all required packages via `!pip install`.
2. Fully executable out of the box with zero external credentials required.
3. Interactive cells demonstrate event emission, time-travel state reconstruction, tamper detection, and what-if branching.

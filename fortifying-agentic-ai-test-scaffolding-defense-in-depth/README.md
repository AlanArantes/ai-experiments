# Fortifying Agentic AI: Building Resilient Test Scaffolding and Defense-in-Depth for LLM Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlanArantes/AI-Experiments/blob/main/fortifying-agentic-ai-test-scaffolding-defense-in-depth/fortifying_agentic_ai_colab_notebook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

Implementation and test scaffolding for the architecture introduced in the article **"Fortifying Agentic AI: Building Resilient Test Scaffolding and Defense-in-Depth for LLM Systems"** by Alan Arantes (Enterprise & System Architect).

---

## 📌 Architectural Overview: Model + Harness

Traditional software testing relies on deterministic inputs and predictable outcomes (`f(x) == y`). Autonomous LLM agents, however, employ non-deterministic multi-turn reasoning and have direct access to execution tools.

Treating the system as **Model + Harness**, this repository establishes engineering-led verification across a **4-Layer Defense-in-Depth Architecture**:

| Defense Layer | Primary Focus | Mechanisms & Tooling |
| :--- | :--- | :--- |
| **1. Foundation Layer** | Deterministic Schemas & Contracts | Pydantic v2 schemas (`eval_models.py`), Atomic Claim decomposition |
| **2. Trajectory Layer** | Execution Telemetry & Safety Scaffolding | Least Privilege enforcement (blocking `drop_table`/`cancel_order`), Loop Prevention (`step_count < MAX_STEPS`), Argument Sanitization |
| **3. Semantic Layer** | Unbiased Evaluators | Deterministic Closed-World LLM-as-a-Judge (`judge_evaluator.py`, OpenAI Structured Outputs / offline auditor) |
| **4. Perimeter Layer** | Adversarial Red-Teaming | Declarative Promptfoo configuration (`promptfooconfig.yaml`), Target Bridge (`agent_tracer_target.py`) |

---

## 📂 Directory Structure

```
fortifying-agentic-ai-test-scaffolding-defense-in-depth/
├── eval_models.py                         # Foundation Layer: Pydantic schemas (ClaimStatus, Verdict, FaithfulnessEvaluation)
├── agent_runtime.py                       # Trajectory Layer: Agent runtime harness, tool permissions, and loop guards
├── judge_evaluator.py                     # Semantic Layer: Closed-World LLM-as-a-Judge implementation
├── promptfooconfig.yaml                   # Perimeter Layer: Declarative Promptfoo red-teaming configuration
├── prompts/
│   └── agent_system_prompt.txt            # System instructions and policy boundaries
├── tests/
│   ├── conftest.py                        # Pytest path setup
│   ├── agent_tracer_target.py             # Promptfoo custom provider target bridge
│   └── test_scaffolding.py                # Comprehensive Pytest test suite (10 test assertions)
├── test_runner.py                         # Interactive CLI demonstration across all 4 layers
├── generate_notebook.py                   # Script generating the Google Colab notebook
├── fortifying_agentic_ai_colab_notebook.ipynb # Fully executable Google Colab notebook
├── pyproject.toml                         # UV project and dependency configuration
└── README.md
```

---

## 🚀 Getting Started with `uv`

### 1. Prerequisites
Ensure [uv](https://docs.astral.sh/uv/) is installed on your system.

### 2. Run the Interactive CLI Demo
```bash
uv run python test_runner.py
```

### 3. Run the Pytest Test Suite
```bash
uv run pytest -v
```

All 10 tests across the 4 layers are executed and validated automatically.

---

## 📓 Running in Google Colab

Open [fortifying_agentic_ai_colab_notebook.ipynb](fortifying_agentic_ai_colab_notebook.ipynb) directly in Google Colab. The notebook is self-contained:
1. Installs all required packages via `!pip install`.
2. Implements and demonstrates each defense layer step-by-step.
3. Operates both with live OpenAI API keys (via OpenAI Structured Outputs) and with a zero-credential deterministic offline auditor fallback for immediate out-of-the-box execution.

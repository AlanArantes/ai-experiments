# AI Experiments Repository

Welcome to the **AI Experiments Repository**! This repository is dedicated to housing and documenting various Jupyter notebooks, code scripts, and resources related to machine learning and artificial intelligence experiments.

## Repository Overview

This repository is structured to help you explore, understand, and replicate AI experiments efficiently. It contains:
- **Jupyter Notebooks**: Step-by-step, well-documented notebooks with code and results from experiments.
- **Python Scripts**: Reusable code components and modules for conducting various machine learning and deep learning tasks.
- **Data and Results**: Sample datasets, generated results, and figures that are referenced in experiments.

The purpose of this repository is to foster learning and allow others to build upon these experiments for research, academic work, or personal projects.

## 🌟 Featured Experiments & Companion Notebooks

| Experiment / Article | Colab Notebook | Local Scaffolding | Focus & Architecture |
| :--- | :--- | :--- | :--- |
| **Fortifying Agentic AI**<br>*(Resilient Test Scaffolding & Defense-in-Depth)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlanArantes/AI-Experiments/blob/main/fortifying-agentic-ai-test-scaffolding-defense-in-depth/fortifying_agentic_ai_colab_notebook.ipynb) | [`fortifying-agentic-ai...`](./fortifying-agentic-ai-test-scaffolding-defense-in-depth/) | 4-layer defense-in-depth: Pydantic schemas, trajectory least-privilege, closed-world LLM-as-a-judge, Promptfoo red-teaming. |
| **Event Sourcing for Agentic AI**<br>*(Deterministic, Auditable & Replayable LLM Systems)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlanArantes/AI-Experiments/blob/main/event-sourcing-for-agentic-ai-deterministic-auditable-replayable/event_sourcing_agentic_ai_colab_notebook.ipynb) | [`event-sourcing-for...`](./event-sourcing-for-agentic-ai-deterministic-auditable-replayable/) | Append-only event store, hash-chain tamper detection, pure state projection fold, time-travel debugging & counterfactual branching. |

## Repository Structure

The repository follows an organized structure for easy navigation:

````markdown
/ai-experiments
  ├── experiment_1/
  │   ├── notebook.ipynb
  │   ├── data/
  │   └── results/
  ├── experiment_2/
  │   ├── notebook.ipynb
  │   ├── model.py
  │   └── utils.py
  └── README.md
````

Each experiment folder typically contains:
- **notebook.ipynb**: Jupyter notebook that explains and executes the experiment.
- **data/**: Folder containing any datasets or data files used in the experiment.
- **results/**: Folder for storing experiment outputs, including model files, visualizations, and performance metrics.
- **utils.py** (optional): Supporting functions, helper scripts, or modules for the experiment.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AlanArantes/AI-Experiments.git
   cd AI-Experiments
2. **Install Dependencies**:
Use the requirements.txt file in the root directory to install all necessary packages.
   ```bash
   pip install -r requirements.txt
3. **Run an Experiment**:
Navigate to an experiment directory and open the notebook in Jupyter:
   ```bash
   jupyter notebook experiments/experiment_1/notebook.ipynb
4. **Customizing and Extending**:
Feel free to modify the code and add new experiments. Each experiment is self-contained, making it easy to extend and create variations.

## Prerequisites
Python 3.7+
Jupyter Notebook
Any additional dependencies are listed in requirements.txt.

## Contributing
Contributions are welcome! Please follow these steps:

**Fork the repository.**
Create a new branch with a descriptive name for your feature or fix.
Submit a pull request with a clear description of changes.

## License
This repository is licensed under the MIT License. See the [MIT License](LICENSE). file for more details.

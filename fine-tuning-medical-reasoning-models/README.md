# Fine-Tuning Medical Reasoning Models

This repository contains a pipeline for fine-tuning Large Language Models on medical Chain-of-Thought (CoT) reasoning datasets using [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl) (Transformer Reinforcement Learning).

---

## 1. Project Overview

- **Pipeline Script**: [`app.py`](file:///C:/Users/alan_/Projetos/labs/ai-experiments/fine-tuning-medical-reasoning-models/app.py)
- **Base Model**: `unsloth/Qwen2.5-3B-Instruct` (scaled for local 6 GB VRAM; can be switched to `14B` for 24 GB+ cloud GPUs)
- **Fine-Tuning Method**: 4-bit QLoRA with rank $r=16$, targeting linear projection modules with gradient checkpointing.
- **Dataset**: `FreedomIntelligence/medical-o1-reasoning-SFT` (English split), formatted into ChatML with reasoning enclosed in `<think>...</think>` tags.
- **Loss Masking**: Trained on assistant responses only via [`train_on_responses_only`](file:///C:/Users/alan_/Projetos/labs/ai-experiments/fine-tuning-medical-reasoning-models/app.py#L124).
- **Environment & Dependency Manager**: [`uv`](https://github.com/astral-sh/uv) via [`pyproject.toml`](file:///C:/Users/alan_/Projetos/labs/ai-experiments/fine-tuning-medical-reasoning-models/pyproject.toml) and [`uv.lock`](file:///C:/Users/alan_/Projetos/labs/ai-experiments/fine-tuning-medical-reasoning-models/uv.lock).

---

## 2. Issues Identified & Diagnostics

### Issue 1: CPU-only PyTorch Installed by Default (`NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.`)

#### Symptoms
Running `uv run python app.py` threw:
```text
Traceback (most recent call last):
  ...
  File "...\unsloth_zoo\device_type.py", line 389, in get_device_type
    raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

#### Root Cause
- On Windows, standard PyPI distributions only host CPU-only wheels for PyTorch (`torch==2.12.1+cpu`).
- As a result, `torch.cuda.is_available()` returned `False` despite an NVIDIA GeForce RTX 4050 Laptop GPU being present.
- Unsloth strictly requires an active GPU accelerator (`cuda`, `hip`, or `xpu`) and cannot run on CPU.

#### Solution
Configured the official PyTorch CUDA index in [`pyproject.toml`](file:///C:/Users/alan_/Projetos/labs/ai-experiments/fine-tuning-medical-reasoning-models/pyproject.toml):
```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
```
Then resolved and synchronized dependencies:
```bash
uv lock
uv sync
```

---

### Issue 2: Unsloth Import Order Warning

#### Symptoms
```text
UserWarning: WARNING: Unsloth should be imported before [trl, transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
```

#### Root Cause
Unsloth patches PyTorch, Hugging Face Transformers, and PEFT internals to apply kernel speedups and activation memory optimizations. Importing `trl`, `transformers`, or `peft` before Unsloth prevents these patches from attaching properly.

#### Solution
Reordered imports at the top of [`app.py`](file:///C:/Users/alan_/Projetos/labs/ai-experiments/fine-tuning-medical-reasoning-models/app.py#L1-L8) so that `unsloth` is imported before `torch`, `datasets`, `trl`, and `transformers`:
```python
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import os
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
```

---

### Issue 3: Local Hardware Constraints (VRAM Limitation)

#### Context
- **Local GPU**: NVIDIA GeForce RTX 4050 Laptop GPU with **6 GB VRAM** (`6141 MiB`).
- **Target Model**: `unsloth/Qwen2.5-14B-Instruct` with `MAX_SEQ_LENGTH = 4096`.

#### VRAM Analysis
- Even with 4-bit quantization, a 14B parameter model requires ~7–8 GB of VRAM solely to store the quantized base model weights.
- When adding LoRA adapter weights, optimizer states (8-bit AdamW), CUDA overhead, and activations for 4096 sequence length, fine-tuning requires a minimum of **16–24 GB VRAM**.
- Running the 14B model locally on a 6 GB GPU will result in a **CUDA Out of Memory (OOM)** error during model loading or training.

#### Recommended Next Steps
1. **For Local Development / Testing (6 GB VRAM)**:
   - Switch `MODEL_NAME` in [`app.py`](file:///C:/Users/alan_/Projetos/labs/ai-experiments/fine-tuning-medical-reasoning-models/app.py#L21) to a smaller model:
     - `unsloth/Qwen2.5-1.5B-Instruct`
     - `unsloth/Qwen2.5-3B-Instruct`
     - `unsloth/Llama-3.2-1B-Instruct`
     - `unsloth/Llama-3.2-3B-Instruct`
   - Reduce `MAX_SEQ_LENGTH` (e.g., to `1024` or `2048`) and `per_device_train_batch_size = 1`.
### Issue 4: MSVC C11 Syntax Error in Triton `driver.c` (`error C2059: erro de sintaxe: '}'`)

#### Symptoms
When initiating the training step, JIT compilation of Triton's CUDA runtime module failed:
```text
C:\...\triton\backends\nvidia\driver.c(1079): error C2059: erro de sintaxe: '}'
C:\...\triton\backends\nvidia\driver.c(1087): error C2059: erro de sintaxe: '}'
subprocess.CalledProcessError: Command '['...\\cl.exe', '...\\driver.c', ... '/std:c11', ...] returned non-zero exit status 2.
```

#### Root Cause
MSVC invoked with `/std:c11` strictly enforces the C standard, where empty struct initializers `= {}` are invalid syntax (unlike C++). Lines 1079 and 1087 of `driver.c` used:
```c
CUlaunchAttribute clusterAttr = {};
CUlaunchAttribute clusterSchedulingAttr = {};
```

#### Solution
Updated both struct initializers in `.venv/Lib/site-packages/triton/backends/nvidia/driver.c` to standard C11 zero-initializers:
```c
CUlaunchAttribute clusterAttr = {0};
CUlaunchAttribute clusterSchedulingAttr = {0};
```
After this patch, Triton JIT compiled `cuda_utils.cp313-win_amd64.pyd` and launched the training kernels without error.

---

## 3. Running the Pipeline

1. **Install / Sync Dependencies**:
   ```bash
   uv sync
   ```

2. **Verify CUDA Availability**:
   ```bash
   uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. **Execute Training**:
   ```bash
   uv run python app.py
   ```

---

## 4. Verification Results

### 1. PyTorch CUDA & Environment Verification
```text
CUDA available: True
Device name: NVIDIA GeForce RTX 4050 Laptop GPU
🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth loaded successfully!
```

### 2. Model Loading Verification (Qwen2.5-3B-Instruct on RTX 4050 6GB)
```text
==((====))==  Unsloth 2026.9.7: Fast Qwen2 patching. Transformers: 5.5.0.
   \\   /|    NVIDIA GeForce RTX 4050 Laptop GPU. Num GPUs = 1. Max memory: 5.997 GB. Platform: Windows.
O^O/ \_/ \    Torch: 2.12.1+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.8.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Loading weights: 100%|██████████| 434/434 [00:03<00:00, 124.03it/s]
Model loaded successfully on GPU!
```

### 3. Training Loop & Triton JIT Kernel Execution
```text
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 19,704 | Num Epochs = 1 | Total steps = 2,463
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 8
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 8 x 1) = 8
 "-____-"     Trainable parameters = 29,933,568 of 3,115,872,256 (0.96% trained)
Loading dataset...
🦥 Unsloth: Padding-free auto-enabled, enabling faster training.
Starting training...
remark: <unknown>:0:0: 106 instructions in function
remark: <unknown>:0:0: 72 instructions in function
remark: <unknown>:0:0: 537 instructions in function
remark: <unknown>:0:0: 309 instructions in function
```

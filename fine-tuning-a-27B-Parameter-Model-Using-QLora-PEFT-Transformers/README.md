# QLoRA Fine-Tuning & vLLM Serving Guide

This document summarizes the pipeline architecture, execution steps, and troubleshooting findings encountered and resolved in this project.

---

## 1. Project Overview

The repository provides an end-to-end LLM fine-tuning and inference workflow:
- **Fine-Tuning (`app.py`)**: Quantized Low-Rank Adaptation (QLoRA) using 4-bit `bitsandbytes`, PEFT, and TRL's `SFTTrainer` on `Qwen/Qwen2.5-3B-Instruct` with the `mlabonne/guanaco-llama2-1k` dataset. Trained adapters are saved to `./qlora-adapter-output/`.
- **Inference Service (`chat.py`)**: Production-ready FastAPI service powered by `vllm.AsyncLLMEngine` supporting real-time Server-Sent Events (SSE) streaming and dynamic runtime LoRA adapter switching.
- **Google Colab Notebooks**:
  - [`qlora_finetuning.ipynb`](./qlora_finetuning.ipynb): **Dedicated 3B Training Notebook (Stops at Step 4)**. Focuses purely on 4-bit SFT fine-tuning, PyTorch evaluation, and adapter export for `Qwen2.5-3B-Instruct` on free T4 GPUs.
  - [`qlora_finetuning_and_vllm_serving.ipynb`](./qlora_finetuning_and_vllm_serving.ipynb): **Full 3B Training + vLLM API Serving Notebook (Steps 1–9)**. Features a two-phase architecture: Phase 1 installs core training packages only and fine-tunes Qwen2.5-3B; Phase 2 cleans GPU memory, separately installs vLLM & API dependencies, launches the background OpenAI-compatible server with dynamic LoRA support (`qlora-adapter`), and runs streaming/non-streaming HTTP requests.
  - [`gemma2_27b_qlora_finetuning.ipynb`](./gemma2_27b_qlora_finetuning.ipynb): **27B Parameter QLoRA Training Notebook**. Fine-tunes `google/gemma-2-27b-it` on `HuggingFaceH4/no_robots` (requires a 24GB+ GPU such as Colab A100 or L4).
    > [!IMPORTANT]
    > **Model Authorization Required**: The `google/gemma-2-27b` and `google/gemma-2-27b-it` models are gated on Hugging Face. You must visit [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) and acknowledge/request access to the license terms (approval is instant), then pass a valid `HF_TOKEN`. If you prefer an ungated ~30B alternative with no authorization step, [`Qwen/Qwen2.5-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) can be used as a drop-in replacement.
- **Package Management**: Managed via [`uv`](https://docs.astral.sh/uv/) using `pyproject.toml` and `uv.lock`.

---

## 2. Setup & Execution Instructions

### Prerequisites
Ensure `uv` is installed and sync the environment:
```bash
uv sync
```

Set up your Hugging Face authentication token from `.env`:
```bash
source .env
```
*(Scripts also automatically load and sanitize `.env` on startup).*

### Step 1: Run Fine-Tuning (`app.py`)
To train or update the LoRA adapter weights:
```bash
uv run python app.py
```
- Loads the base model in 4-bit NormalFloat quantization.
- Attaches LoRA adapters (`r=16`, `lora_alpha=32`) to linear attention projections.
- Saves adapter weights and tokenizer to `./qlora-adapter-output/`.

### Step 2: Run the vLLM Inference Service (`chat.py`)
To launch the FastAPI streaming server:
```bash
uv run python chat.py
```
Alternatively, using Uvicorn directly:
```bash
uv run uvicorn chat:app --host 0.0.0.0 --port 8000
```

#### Running via `python` vs. `uvicorn` CLI
Both approaches run the exact same underlying Uvicorn ASGI server because `chat.py` invokes `uvicorn.run(...)` programmatically when executed as `__main__`.

| Feature | `uv run python chat.py` | `uv run uvicorn chat:app` |
| :--- | :--- | :--- |
| **Convenience** | **Recommended**: Self-contained. No need to specify host or port flags. | Requires typing `--host 0.0.0.0 --port 8000`. |
| **CLI Flexibility** | Host and port are defined in code. | **Better for flags**: Easily override parameters (`--port 8080`, `--log-level debug`, etc.). |
| **Environment Sanitization** | Guarantees top-level `.env` loading and token cleanup runs before server init. | Module imported dynamically by CLI. |
| **Debugging** | Standard Python process; direct support for `breakpoint()` and profilers. | Wrapped by the Uvicorn CLI runner. |

> [!WARNING]
> **Never use `--workers > 1` with vLLM on a single GPU**: Each worker process spawns an independent copy of `AsyncLLMEngine`. On a 6 GB GPU, multiple workers will immediately trigger a CUDA Out-of-Memory (OOM) error. Keep workers set to 1.


### Step 3: Querying the Service

- **Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```

- **Base Model Query (Standard JSON)**:
  ```bash
  curl -X POST http://localhost:8000/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "What are the benefits of QLoRA fine-tuning?",
      "max_tokens": 128
    }'
  ```

- **Fine-Tuned Adapter Query (Standard JSON)**:
  ```bash
  curl -X POST http://localhost:8000/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "What are the benefits of QLoRA fine-tuning?",
      "model": "qlora-adapter",
      "max_tokens": 128
    }'
  ```

- **Streaming Query (`"stream": true`)**:
  ```bash
  curl -N -X POST http://localhost:8000/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "What are the benefits of QLoRA fine-tuning?",
      "model": "qlora-adapter",
      "stream": true,
      "max_tokens": 128
    }'
  ```

#### Using HTTPie (`http`)
HTTPie sends JSON automatically with `key=value` for strings and `key:=value` for numbers/booleans:

- **Health Check**:
  ```bash
  http :8000/health
  ```

- **Standard Complete JSON Response (Default `stream: false`)**:
  ```bash
  http POST :8000/v1/generate \
    prompt="What are the benefits of QLoRA fine-tuning?" \
    model="qlora-adapter" \
    max_tokens:=128
  ```
  *Response format:*
  ```json
  {
    "id": "gen-a1b2c3d4",
    "model": "qlora-adapter",
    "text": "QLoRA reduces memory usage by quantizing base weights...",
    "finish_reason": "stop",
    "usage": {
      "prompt_tokens": 14,
      "completion_tokens": 42,
      "total_tokens": 56
    }
  }
  ```

- **Live Streaming SSE Response (`stream:=true`)**:
  ```bash
  http --stream POST :8000/v1/generate \
    prompt="What are the benefits of QLoRA fine-tuning?" \
    model="qlora-adapter" \
    stream:=true \
    max_tokens:=128
  ```

*(Note: `/generate` is also supported as an alias for convenience).*


---

## 3. Findings & Troubleshooting Log

During runtime execution, three compatibility issues were identified and resolved:

### Issue 1: `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
- **Symptom**: `chat.py` crashed during startup when vLLM called `tokenizer.all_special_tokens_extended`.
- **Root Cause**: `pyproject.toml` specified `transformers>=4.45.0` without an upper bound constraint. `uv` installed `transformers 5.17.0`. Hugging Face v5 introduced breaking changes to tokenizer internals and removed `all_special_tokens_extended`, which `vllm 0.8.5` requires.
- **Fix**: Pinned `transformers>=4.45.0,<5.0.0` in `pyproject.toml` and ran `uv sync` to install `transformers 4.57.6`.

### Issue 2: `ModuleNotFoundError: No module named 'setuptools'`
- **Symptom**: vLLM worker initialization failed inside `triton/runtime/build.py`.
- **Root Cause**: `triton` dynamically imports `setuptools` during backend driver discovery. Standard `uv` virtual environments under Python 3.11 do not include `setuptools` by default.
- **Fix**: Added `"setuptools"` to `dependencies` in `pyproject.toml` and ran `uv sync`.

### Issue 3: `requests.exceptions.InvalidHeader: Invalid leading whitespace, reserved character(s), or return character(s) in header value: 'Bearer ...\r'`
- **Symptom**: vLLM failed while querying model repository metadata on Hugging Face Hub.
- **Root Cause**: The `.env` file had Windows CRLF line endings (`\r\n`). Exporting via `export $(cat .env | xargs)` included the trailing `\r` into the shell's `$HF_TOKEN` environment variable. RFC HTTP header validation in `requests` rejected the carriage return.
- **Fix**:
  1. Converted `.env`, `app.py`, and `chat.py` line terminators from CRLF to LF (`sed -i 's/\r$//'`).
  2. Added an automated token-sanitization block to the entry point of both `chat.py` and `app.py`:
     ```python
     import os
     from dotenv import load_dotenv

     load_dotenv()
     if "HF_TOKEN" in os.environ:
         os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"].strip()
     ```

### Issue 4: `fatal error: stdlib.h: No such file or directory` (Triton JIT compilation)
- **Symptom**: During model profiling / KV cache initialization, GCC failed while compiling Triton's `cuda_utils` backend driver.
- **Root Cause**: On Ubuntu, the `gcc` compiler package was installed without standard C library development headers (`libc6-dev` / `build-essential`). When Triton invokes `gcc` to build its runtime C extension, `#include <stdlib.h>` fails.
- **Fix**: Install the standard development headers on your system:
  ```bash
  sudo apt update && sudo apt install -y build-essential
  ```

### Issue 5: `ValueError: No available memory for the cache blocks` (KV-Cache OOM)
- **Symptom**: Engine startup aborted with `No available memory for the cache blocks. Try increasing gpu_memory_utilization`.
- **Root Cause**:
  1. In unquantized `bfloat16`, Qwen2.5-3B weights alone take 5.8 GB, leaving zero VRAM on a 6 GB card.
  2. The default context length (`max_model_len`) defaulted to the full 32,768 tokens, demanding >1.2 GB per sequence.
  3. Inductor/Dynamo compilation captured large CUDA graph pools and took 67s on startup.
- **Fix**: Updated `AsyncEngineArgs` in `chat.py`:
  - `quantization="bitsandbytes"` and `load_format="bitsandbytes"` (shrinks model footprint to ~2.0 GB).
  - `max_model_len=2048` (reduces sequence reservation).
  - `enforce_eager=True` (skips 67s `torch.compile` and eliminates graph memory pools).
  - `max_loras=1` and `max_lora_rank=16`.
  - **Result**: Frees 17,152 tokens of KV cache (8.38x concurrency) with instant ~8s startup.

### Issue 6: `ImportError: libcudart.so.13: cannot open shared object file: No such file or directory`
- **Symptom**: In Google Colab or CUDA 12 environments, importing vLLM (`from vllm import AsyncLLMEngine`) fails when loading `vllm._C` with missing `libcudart.so.13`.
- **Root Cause**: Installing `vllm` without an upper bound (`vllm>=0.6.0`) pulls the latest release (`vllm>=0.20.0`). The default PyPI wheels for newer vLLM releases are compiled against CUDA 13.0 (`libcudart.so.13`). Standard Google Colab GPU runtimes provide CUDA 12 (`libcudart.so.12`).
- **Fix**:
  - In your environment/notebook, pin `vllm>=0.6.0,<=0.8.5.post1` (or `vllm==0.8.5.post1`), which natively links to `libcudart.so.12` and remains compatible with `transformers<5.0.0` and `torch==2.6.0`.
  - To recover an active Colab session without losing trained adapter weights:
    ```bash
    !pip uninstall -y vllm
    !pip install "vllm==0.8.5.post1"
    ```
    Then restart the session (`Runtime -> Restart session`) and run directly from Step 5.

### Issue 7: `TemplateError` in Gemma 2 Chat Template (`System role not supported` / `Roles must alternate`)
- **Symptom**: During dataset preprocessing (`dataset.map(format_conversations)`), Jinja2 raises:
  - `TemplateError: System role not supported` when encountering `"role": "system"`.
  - `TemplateError: Conversation roles must alternate user/assistant/user/assistant/...` when encountering consecutive assistant or user turns (e.g., sample #487 in `no_robots`).
- **Root Cause**: The official Google Gemma 2 Jinja chat template enforces strict state machine rules: only `"user"` and `"model"` roles are permitted, and each turn must strictly alternate (`user` $\rightarrow$ `model` $\rightarrow$ `user` $\rightarrow$ `model`).
- **Fix**:
  1. Prepend any `"system"` prompts into the first `"user"` message (`system_prompt + content`).
  2. Map `"assistant"` to `"model"`.
  3. Merge consecutive turns with the same role (`merged[-1]["content"] += "\n\n" + turn["content"]`).

### Issue 8: `KeyError: 'completion'` in SFTTrainer
- **Symptom**: `SFTTrainer` fails during dataset initialization with `KeyError: 'completion'`.
- **Root Cause**: The raw `HuggingFaceH4/no_robots` dataset contains columns `['prompt', 'prompt_id', 'messages', 'category']`. When `raw_dataset.map()` created a new `"text"` column without `remove_columns`, the old `"prompt"` column remained in the dataset. TRL's `SFTTrainer` checks `if "prompt" in example:`, assumes the dataset is in Prompt-Completion format, and tries to index `example["completion"]`.
- **Fix**: Add `remove_columns=raw_dataset.column_names` to `raw_dataset.map()`. This drops all legacy columns and leaves only the single formatted `"text"` column for `dataset_text_field="text"`.

### Issue 9: Pip Hangs / Silent Infinite Compilation during vLLM Installation
- **Symptom**: In Step 5, `pip install` downloads ~3.5 GB of wheels, and then the cell appears to hang indefinitely without producing any error messages or progress output.
- **Root Cause**: 
  1. Older pinned versions like `vllm<=0.8.5.post1` do not have pre-compiled binary wheels (`.whl`) for Python 3.13 on PyPI (they only support up to Python 3.12).
  2. Because no binary wheel is found, `pip` falls back to building from source (`setup.py` / `pyproject.toml`), downloading the entire C++/CUDA developer toolchain and compiling vLLM's kernels from scratch (which takes 45+ minutes on cloud CPU instances).
  3. The `-q` (quiet) flag suppresses compiler logs, making pip look completely frozen.
- **Fix**: Use `uv` with prebuilt binary wheels (`!pip install -q uv && uv pip install --system vllm fastapi "uvicorn[standard]" requests openai`). `uv` resolves modern prebuilt wheels directly for the active Python version and installs in under 45 seconds without compiling from source.

### Issue 10: `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions`
- **Symptom**: The vLLM server fails to start with:
  ```
  RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA version 12.8.
  ```
- **Root Cause**: Google Colab comes with `torchaudio` preinstalled, compiled for CUDA 12.8. When `vllm` is installed, it updates `torch` to CUDA 13.0. During server initialization, `transformers` transitively imports `torchaudio` in `audio_utils.py`, triggering `torchaudio`'s version assertion check. Since Qwen2.5 is a pure text LLM, audio processing is not used.
- **Fix**: Run `!pip uninstall -y torchaudio`. Once uninstalled, `transformers` catches the expected `ImportError`, sets `_torchaudio_available = False`, and vLLM boots cleanly without crashing.

### Issue 11: `Value error, Unknown quantization method: bitsandbytes` in vLLM
- **Symptom**: vLLM crashes during `ModelConfig` validation with:
  ```
  pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Unknown quantization method: bitsandbytes. Must be one of ['awq', 'auto_awq', 'fp8', ... 'torchao', ...]
  ```
  Additionally, `--disable-log-requests` is reported as an unrecognized CLI flag in modern vLLM releases.
- **Root Cause**: Recent vLLM releases deprecated the legacy `bitsandbytes` quantization backend in favor of `torchao`, AWQ, and FP8.
- **Fix**: Remove `--quantization bitsandbytes`, `--load-format bitsandbytes`, and `--disable-log-requests`. Use `--dtype auto` instead. For a 3B model like Qwen2.5-3B, unquantized 16-bit weights consume only ~6.0 GB of VRAM, fitting effortlessly on Google Colab GPUs (T4: 15 GB, L4: 24 GB, A100: 80 GB) while delivering significantly faster generation speeds without on-the-fly dequantization latency.

### Issue 12: `localtunnel` Hanging on Installation Prompt & HTML Reminder Page
- **Symptom**: 
  1. Running `!npx localtunnel --port 8000` hangs indefinitely waiting for keyboard input.
  2. Sending `curl` requests to the public `loca.lt` URL returns an HTML page ("Friendly Reminder...") instead of JSON responses from the vLLM server.
- **Root Cause**: 
  1. `npx` prompts interactively (`Need to install the following packages: localtunnel... Ok to proceed? (y)`). Without `-y`, the non-interactive Colab cell blocks indefinitely.
  2. `localtunnel` inserts an anti-abuse reminder screen on all incoming HTTP requests unless a bypass header is supplied.
- **Fix**: 
  1. Run `!npx -y localtunnel --port 8000`.
  2. Include `-H "Bypass-Tunnel-Reminder: true"` in all `curl` and Python API requests.

---

## 4. Hardware & Memory Notes

- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM).
- **Training**: 4-bit NF4 quantization keeps VRAM usage at ~2–3 GB during fine-tuning with `per_device_train_batch_size=2` and `gradient_accumulation_steps=4`.
- **Inference**: Configured with 4-bit bitsandbytes quantization and eager execution, loading the base model in ~2.0 GB VRAM with 17,152 tokens of KV cache available for streaming generation.

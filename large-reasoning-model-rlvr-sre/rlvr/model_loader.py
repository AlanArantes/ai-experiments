"""
Model & Tokenizer Loader for Qwen-2.5-Coder and Llama Base Reasoning Models.
Loads the active policy model, frozen reference model, and tokenizer for GRPO / SFT training.
"""

import torch

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
LIGHTWEIGHT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"

def load_reasoning_models(
    model_id: str = DEFAULT_MODEL_ID,
    device_map: str = "auto",
    dtype: torch.dtype | None = None,
    load_ref_model: bool = True
):
    """
    Loads:
      1. Tokenizer with chat template and proper padding configuration.
      2. Policy Model (active, trainable weights).
      3. Reference Model (frozen copy required for GRPO KL penalty).
    """
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[*] Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading policy model '{model_id}' (dtype={dtype}, device_map={device_map})...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    ref_model = None
    if load_ref_model:
        print(f"[*] Loading reference model '{model_id}' (frozen copy for GRPO KL penalty)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    return policy_model, ref_model, tokenizer

if __name__ == "__main__":
    print(f"Default 7B Model: {DEFAULT_MODEL_ID}")
    print(f"Lightweight Local CPU Alternative: {LIGHTWEIGHT_MODEL_ID}")

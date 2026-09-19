import os
from dotenv import load_dotenv

# Load and sanitize environment variables
load_dotenv()
if "HF_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"].strip()

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# 1. Base Configuration & Model Identifiers
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Completely open / ungated (Apache 2.0)
DATASET_ID = "mlabonne/guanaco-llama2-1k"
OUTPUT_DIR = "./qlora-adapter-output"

# ---------------------------------------------------------------------------
# 2. Tokenizer Setup
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------------------------------------------------------------------------
# 3. 4-bit NormalFloat Quantization (bitsandbytes)
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# ---------------------------------------------------------------------------
# 4. Load Base Model & Prepare for k-bit Training
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# Freezes base weights and casts layernorms/lm_head for gradient stability
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Cache must be disabled for gradient checkpointing

# ---------------------------------------------------------------------------
# 5. PEFT Adapter Config (LoRA)
# ---------------------------------------------------------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Targets all standard linear projection blocks in modern transformer architectures
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ---------------------------------------------------------------------------
# 6. Load Dataset
# ---------------------------------------------------------------------------
dataset = load_dataset(DATASET_ID, split="train")

# ---------------------------------------------------------------------------
# 7. SFTTrainer & SFTConfig Training Setup
# ---------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",       # Offloads optimizer states to prevent OOM spikes
    gradient_checkpointing=True,    # Recomputes activations during backward pass
    report_to="none",               # Set to "wandb" or "tensorboard" if tracking
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# ---------------------------------------------------------------------------
# 8. Train and Persist Only the Adapter Weights
# ---------------------------------------------------------------------------
trainer.train()

# Saves only the LoRA adapter (~50MB-200MB) rather than full model weights
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete. LoRA adapter saved to {OUTPUT_DIR}")

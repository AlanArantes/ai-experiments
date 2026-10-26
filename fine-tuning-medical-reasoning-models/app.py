import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import os
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

load_dotenv()
if "HF_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"].strip()

# ---------------------------------------------------------------------------
# 1. Configuration & Model Initialization
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 2048  # Optimized context length to fit within 6GB VRAM
DTYPE = None  # None = auto-detect (Float16 or Bfloat16)
LOAD_IN_4BIT = True  # 4-bit QLoRA to fit comfortably on 6GB VRAM

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# ---------------------------------------------------------------------------
# 2. Configure PEFT / LoRA Adapters
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Unsloth is optimized for lora_dropout=0
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves 30%+ activation VRAM
    random_state=3407,
)

# ---------------------------------------------------------------------------
# 3. Format Data with Reasoning Separation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert clinical reasoning assistant. Think step-by-step through "
    "the differential diagnosis, pathophysiology, and clinical evidence before "
    "providing the final answer."
)


def format_medical_cot(batch):
    formatted_texts = []
    for question, cot, response in zip(
        batch["Question"], batch["Complex_CoT"], batch["Response"]
    ):
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{cot}\n</think>\n{response}<|im_end|>"
        )
        formatted_texts.append(text)
    return {"text": formatted_texts}


print("Loading dataset...")
raw_dataset = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "en",  # English split; use "zh" or "en_mix" as needed
    split="train",
)

formatted_dataset = raw_dataset.map(
    format_medical_cot,
    batched=True,
    remove_columns=raw_dataset.column_names,
)

# ---------------------------------------------------------------------------
# 4. Training Arguments & SFTTrainer Execution
# ---------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir="./outputs_medical_o1_qwen3b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    warmup_ratio=0.05,
    num_train_epochs=1,  # 1 epoch is standard for reasoning datasets
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
)

# ---------------------------------------------------------------------------
# 5. Loss Masking (Train on Responses Only)
# ---------------------------------------------------------------------------
# Computes loss strictly on `<|im_start|>assistant\n` response tokens.
# User prompt & system prompt tokens are masked out (-100).
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print("Starting training...")
trainer_stats = trainer.train()

# ---------------------------------------------------------------------------
# 6. Save LoRA Adapters and GGUF/Merged Weights
# ---------------------------------------------------------------------------
model.save_pretrained("./medical_o1_lora_model")
tokenizer.save_pretrained("./medical_o1_lora_model")

# Optional: Export directly to 16-bit merged weights or GGUF
# model.save_pretrained_merged("./medical_o1_merged_16bit", tokenizer, save_method="merged_16bit")

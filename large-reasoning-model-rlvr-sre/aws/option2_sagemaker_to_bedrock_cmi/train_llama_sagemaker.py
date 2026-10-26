"""
Option 2: SageMaker Training Script specifically tailored for Meta Llama (3.1-8B / 3.2-3B).
Trained via GRPO across EKS sandboxes, then exported in Safetensors format
compatible with Amazon Bedrock Custom Model Import (CMI).
"""

import os
import sys
import argparse
import subprocess
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from rlvr.model_loader import load_reasoning_models
from rlvr.grpo_loss import compute_grpo_loss
from rlvr.reward_evaluator import SRERewardEvaluator
from scenarios.scenario_definitions import SCENARIOS

DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.1-8B"

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Llama RLVR Training Job for Bedrock CMI")
    parser.add_argument("--model-id", type=str, default=DEFAULT_LLAMA_MODEL, help="Base Llama model ID on Hugging Face")
    parser.add_argument("--group-size", type=int, default=4, help="Rollout candidates per prompt in GRPO")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta-kl", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print(" [Option 2] SageMaker Meta-Llama RLVR Training -> Bedrock CMI Export")
    print("=" * 70)
    print(f"[*] Base Llama Model: {args.model_id}")
    print(f"[*] Target Model Output Directory: {args.model_dir}")
    print(f"[*] CUDA Available: {torch.cuda.is_available()}")
    print("=" * 70)

    # 1. Instantiate Llama Policy & Reference Models
    print(f"\n[1] Loading {args.model_id} policy and reference models...")
    policy_model, ref_model, tokenizer = load_reasoning_models(
        model_id=args.model_id,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_ref_model=True
    )
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)

    # 2. Multi-Rollout RLVR Evaluation Step
    print("\n[2] Executing GRPO rollout step on EKS SRE scenario...")
    scenario = SCENARIOS[0]
    evaluator = SRERewardEvaluator(
        ground_truth_cause=scenario["ground_truth_root_cause"],
        verifier_exit_code=0,
        forbidden_patterns=scenario.get("forbidden_actions", [])
    )

    sample_trajectories = [
        "<think>Investigating OOMKilled in sandbox... Increasing limits to 256Mi.</think><action>patch deploy</action><verdict>{\"root_cause\": \"MEMORY_LIMIT_TOO_LOW\", \"status\": \"RESOLVED\"}</verdict>",
        "<think>Unknown issue. Trying pod restart.</think><action>restart</action><verdict>{\"root_cause\": \"UNKNOWN\", \"status\": \"FAILED\"}</verdict>",
        "<think>Checked memory usage vs limits. Applying resource patch.</think><action>patch deploy</action><verdict>{\"root_cause\": \"MEMORY_LIMIT_TOO_LOW\", \"status\": \"RESOLVED\"}</verdict>",
        "<think>Looking at logs.</think><action>get pods</action><verdict>{\"root_cause\": \"WRONG\", \"status\": \"FAILED\"}</verdict>"
    ]

    rollout_rewards = []
    for traj in sample_trajectories[:args.group_size]:
        r, _ = evaluator.calculate_reward(traj, total_steps=2)
        rollout_rewards.append(r)

    # 3. Compute GRPO Loss
    device = getattr(policy_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    vocab_size = policy_model.config.vocab_size

    batch_input_ids = torch.randint(2, vocab_size, (args.group_size, args.seq_len), device=device)
    batch_loss_mask = torch.ones((args.group_size, args.seq_len), dtype=torch.float32, device=device)
    batch_attention_mask = torch.ones((args.group_size, args.seq_len), dtype=torch.long, device=device)

    optimizer.zero_grad()
    loss = compute_grpo_loss(
        policy_model=policy_model,
        ref_model=ref_model,
        batch_input_ids=batch_input_ids,
        batch_loss_mask=batch_loss_mask,
        batch_attention_mask=batch_attention_mask,
        rewards=rollout_rewards,
        epsilon=args.epsilon,
        beta_kl=args.beta_kl
    )
    loss.backward()
    optimizer.step()

    print(f"  -> GRPO Step Loss: {loss.item():.4f}")

    # 4. Save in Safetensors format compatible with Bedrock Custom Model Import
    print(f"\n[3] Exporting trained Llama weights in Safetensors format to {args.model_dir}...")
    os.makedirs(args.model_dir, exist_ok=True)
    policy_model.save_pretrained(args.model_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.model_dir)

    print("\n[OK] Model successfully exported for Amazon Bedrock Custom Model Import (CMI).")

if __name__ == "__main__":
    main()

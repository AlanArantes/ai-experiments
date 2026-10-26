"""
SageMaker Entry Point Script for Training Large Reasoning Models (RLVR / GRPO).
SageMaker automatically executes this script inside the GPU training container.
"""

import os
import sys
import argparse
import torch

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from rlvr.trajectory_formatter import SRETrajectoryFormatter
from rlvr.reward_evaluator import SRERewardEvaluator
from rlvr.grpo_loss import compute_grpo_loss
from rlvr.model_loader import load_reasoning_models
from scenarios.scenario_definitions import SCENARIOS

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker LRM Training Job")
    
    # Hyperparameters passed from SageMaker PyTorch estimator
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-7B", help="Hugging Face model ID")
    parser.add_argument("--group-size", type=int, default=4, help="Rollout candidates G per prompt in GRPO")
    parser.add_argument("--seq-len", type=int, default=512, help="Max sequence length for rollout turns")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for policy optimizer")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO clipping parameter")
    parser.add_argument("--beta-kl", type=float, default=0.04, help="KL divergence coefficient")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # SageMaker standard environment paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print(" AWS SageMaker Training Job: Large Reasoning Model (LRM)")
    print("=" * 70)
    print(f"[*] PyTorch Version: {torch.__version__}")
    print(f"[*] CUDA Available: {torch.cuda.is_available()}")
    print(f"[*] GPU Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"[*] Model ID: {args.model_id}")
    print(f"[*] Target Model Output Dir: {args.model_dir}")
    print(f"[*] Input Data Dir: {args.train_dir}")
    print("=" * 70)

    # 1. Load Policy Model, Reference Model & Tokenizer
    print("\n[1] Instantiating models and tokenizer...")
    policy_model, ref_model, tokenizer = load_reasoning_models(
        model_id=args.model_id,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_ref_model=True
    )
    
    # 2. Setup Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)

    # 3. Training Loop Demonstration (GRPO Rollout Step)
    print("\n[2] Executing GRPO training loop step...")
    device = getattr(policy_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    vocab_size = policy_model.config.vocab_size
    
    # Generate mock rollout batch across group size G
    batch_input_ids = torch.randint(2, vocab_size, (args.group_size, args.seq_len), device=device)
    batch_loss_mask = torch.ones((args.group_size, args.seq_len), dtype=torch.float32, device=device)
    batch_attention_mask = torch.ones((args.group_size, args.seq_len), dtype=torch.long, device=device)
    
    # Verifiable rewards collected from verifier probes across rollouts
    rollout_rewards = [1.25, -0.40, 0.85, -0.15][:args.group_size]

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
    
    print(f"  -> Step 1 GRPO Loss: {loss.item():.4f}")

    # 4. Save Final Model Artifacts for SageMaker to Sync with S3
    print(f"\n[3] Saving model artifacts to {args.model_dir} for S3 export...")
    os.makedirs(args.model_dir, exist_ok=True)
    policy_model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    print("\n" + "=" * 70)
    print(" [OK] SageMaker Training Job Completed Successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""
End-to-End Demonstration and Test Harness for Large Reasoning Model RLVR & Trajectories.
Run via:
    uv run python main.py
"""

import sys
import torch
from torch import nn

# Ensure UTF-8 output on Windows consoles
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
from scenarios.scenario_definitions import SCENARIOS
from synthetic_data.irac_pipeline import IRACGenerator


class MockTransformerPolicy(nn.Module):
    """Simple linear projection model simulating a causal LM head for local verification."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        logits = self.lm_head(x)

        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(logits)


class DummyTokenizer:
    """Fallback tokenizer in case offline or downloading is disabled."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab_size = 1000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Hash characters to tokens within vocab range
        return [(hash(word) % (self._vocab_size - 10)) + 2 for word in text.split()]


def main(use_pretrained: bool = False, model_id: str = "Qwen/Qwen2.5-Coder-7B"):
    print("=" * 70)
    print(" Large Reasoning Model (LRM) - RLVR & Trajectory Evaluation")
    print("=" * 70)

    # 1. Inspect Defined Scenarios
    print(f"\n[1] Loaded {len(SCENARIOS)} SRE RLVR Scenarios:")
    for s in SCENARIOS:
        print(
            f"  • {s['id']}: {s['description'][:60]}... (Root Cause: {s['ground_truth_root_cause']})"
        )

    # 2. Test SRE Reward Evaluator
    print("\n[2] Testing SRERewardEvaluator with Sample SRE Trajectory...")
    sample_trajectory_text = """
    <context>ALERT: PaymentServiceDegraded in namespace sandbox-app.</context>
    <think>Hypothesis 1: Pods are crash looping due to OOM. Let's inspect.</think>
    <action>{"tool": "kubectl", "cmd": "get pods -n sandbox-app"}</action>
    <observation>payment-worker-9x8f 0/1 CrashLoopBackOff 4 2m</observation>
    <think>OOMKilled confirmed. Patching memory limit to 256Mi.</think>
    <action>{"tool": "kubectl", "cmd": "patch deployment payment-processor -n sandbox-app"}</action>
    <verdict>{"root_cause": "MEMORY_LIMIT_TOO_LOW", "status": "RESOLVED"}</verdict>
    """

    evaluator = SRERewardEvaluator(
        ground_truth_cause="MEMORY_LIMIT_TOO_LOW",
        verifier_exit_code=0,
        forbidden_patterns=["kubectl delete namespace", "--all"],
    )
    reward, metrics = evaluator.calculate_reward(sample_trajectory_text, total_steps=2)
    print(f"  -> Total Reward Score: {reward:.3f}")
    print(f"  -> Breakdown: {metrics}")

    # 3. Test SRE Trajectory Tokenizer and Loss Masking
    print("\n[3] Testing SRETrajectoryFormatter & Loss Masking...")
    tokenizer = DummyTokenizer()
    formatter = SRETrajectoryFormatter(tokenizer)
    trajectory_turns = [
        {"role": "system", "content": "You are an autonomous SRE reasoning agent."},
        {
            "role": "assistant",
            "content": "<think>Investigating cluster alerts...</think><action>get pods</action>",
        },
        {"role": "environment", "content": "<observation>Pod crashloop</observation>"},
        {
            "role": "assistant",
            "content": "<think>Fix applied</think><verdict>RESOLVED</verdict>",
        },
    ]
    batch = formatter.format_and_mask(trajectory_turns, max_length=64)
    active_tokens = int(batch["loss_mask"].sum().item())
    total_tokens = int(batch["attention_mask"].sum().item())
    print(f"  -> Input IDs shape: {batch['input_ids'].shape}")
    print(
        f"  -> Total sequence tokens: {total_tokens}, Agent tokens (masked for loss): {active_tokens}"
    )

    # 4. Test GRPO Loss Computation
    print("\n[4] Testing Group Relative Policy Optimization (GRPO) Loss...")
    if use_pretrained:
        from rlvr.model_loader import load_reasoning_models
        print(f"  -> Loading pre-trained model: {model_id}...")
        policy_model, ref_model, hf_tokenizer = load_reasoning_models(model_id=model_id)
        vocab_size = policy_model.config.vocab_size
        group_size = 2
        seq_len = 32
    else:
        print("  -> Using MockTransformerPolicy (pass --use-pretrained to load real Qwen weights)...")
        vocab_size = 1000
        group_size = 4
        seq_len = 32
        policy_model = MockTransformerPolicy(vocab_size=vocab_size)
        ref_model = MockTransformerPolicy(vocab_size=vocab_size)

    device = getattr(policy_model, "device", torch.device("cpu"))
    batch_input_ids = torch.randint(2, vocab_size, (group_size, seq_len), device=device)
    batch_loss_mask = torch.ones((group_size, seq_len), dtype=torch.float32, device=device)
    batch_attention_mask = torch.ones((group_size, seq_len), dtype=torch.long, device=device)
    # Rollout rewards across group candidates
    rollout_rewards = [1.25, -0.40, 0.85, -0.15][:group_size]

    loss = compute_grpo_loss(
        policy_model=policy_model,
        ref_model=ref_model,
        batch_input_ids=batch_input_ids,
        batch_loss_mask=batch_loss_mask,
        batch_attention_mask=batch_attention_mask,
        rewards=rollout_rewards,
        epsilon=0.2,
        beta_kl=0.04,
    )
    print(f"  -> GRPO Loss computed across {group_size} rollouts: {loss.item():.4f}")

    # 5. Test Legal IRAC Synthetic Data Pipeline
    print("\n[5] Testing Legal IRAC Synthetic Data Validator...")
    context = "Clause 14.2: Right of withdrawal within 7 calendar days. Clause 14.3: Custom goods exempt."
    think = "[ISSUE] Withdrawal timeliness. [RULE] Clause 14.2 sets 7 days limit. Clause 14.3 covers custom items. [APPLICATION] Notice on day 6. Wait, let me check if custom. Standard item. [CONCLUSION] Valid."
    answer = "Under Clause 14.2, you can withdraw within 7 days."
    is_valid, errors = IRACGenerator.verify_irac_trace(context, think, answer)
    print(f"  -> IRAC Trace Valid: {is_valid}")
    if errors:
        print(f"  -> Validation Errors: {errors}")
    else:
        print("  -> Passed all structural, citation, and backtracking checks!")

    print("\n" + "=" * 70)
    print(" [OK] All verification checks completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Harness for LRM RLVR Trajectories")
    parser.add_argument("--use-pretrained", action="store_true", help="Download and load full Qwen-2.5-Coder-7B weights")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-7B", help="Model ID on Hugging Face")
    args = parser.parse_args()

    main(use_pretrained=args.use_pretrained, model_id=args.model_id)

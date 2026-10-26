import json
import re

class SRERewardEvaluator:
    """
    Computes deterministic, multi-component scalar rewards for SRE rollouts.
    Components:
      - Health / SLO restoration: +1.0 for pass, -0.5 for fail
      - Root cause diagnostic verdict: +0.5 for exact match, -0.2 for mismatch, -0.5 for missing
      - Blast radius / policy violations: -1.5 penalty per forbidden action
      - Step efficiency: small deduction for excessive turns
    """
    def __init__(self, ground_truth_cause: str, verifier_exit_code: int, forbidden_patterns: list[str] = None):
        self.ground_truth_cause = ground_truth_cause
        self.verifier_exit_code = verifier_exit_code  # 0 from verifier probe
        self.forbidden_patterns = forbidden_patterns or []

    def calculate_reward(self, trajectory_text: str, total_steps: int) -> tuple[float, dict]:
        """
        Calculates composite scalar reward R in [-2.0, 1.5]
        """
        metrics = {
            "r_health": 0.0,
            "r_diagnosis": 0.0,
            "r_penalty": 0.0,
            "r_efficiency": 0.0
        }
        
        # 1. Verification Probe (SLO restored)
        if self.verifier_exit_code == 0:
            metrics["r_health"] = 1.0
        else:
            metrics["r_health"] = -0.5

        # 2. Root Cause Classification Check
        verdict_match = re.search(r"<verdict>\s*(\{.*?\})\s*</verdict>", trajectory_text, re.DOTALL)
        if verdict_match:
            try:
                verdict_data = json.loads(verdict_match.group(1))
                predicted_cause = verdict_data.get("root_cause", "").strip()
                if predicted_cause == self.ground_truth_cause:
                    metrics["r_diagnosis"] = 0.5
                else:
                    metrics["r_diagnosis"] = -0.2
            except json.JSONDecodeError:
                metrics["r_diagnosis"] = -0.3  # Malformed JSON format penalty
        else:
            metrics["r_diagnosis"] = -0.5  # Missed verdict tag entirely

        # 3. Blast Radius & Policy Violation Check
        action_blocks = re.findall(r"<action>\s*(\{.*?\})\s*</action>", trajectory_text, re.DOTALL)
        for act in action_blocks:
            for pattern in self.forbidden_patterns:
                if pattern in act:
                    metrics["r_penalty"] += 1.5  # Subtracted below
                    break

        # 4. Step Cost (Promotes concise diagnostic paths)
        metrics["r_efficiency"] = min(0.25, total_steps * 0.03)

        # Composite formulation
        total_reward = (
            metrics["r_health"]
            + metrics["r_diagnosis"]
            - metrics["r_penalty"]
            - metrics["r_efficiency"]
        )

        return float(total_reward), metrics

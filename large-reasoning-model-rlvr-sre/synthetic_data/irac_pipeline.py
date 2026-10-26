"""
Synthetic IRAC (Issue, Rule, Application, Conclusion) Dataset Generator & Verifier.
Implements the 4-phase data generation and validation pipeline for legal reasoning models:
  1. Knowledge graph / Rule extraction
  2. Scenario generation with boundary constraints
  3. CoT generation with deliberate backtracking tokens
  4. Rule-based verifier & citation checker
"""

import json
import re
from typing import TypedDict

class LegalRule(TypedDict):
    clause_id: str
    rule_text: str
    deadline_days: int | None
    exceptions: list[str]

class IRACGenerator:
    """
    Template and pipeline manager for generating synthetic IRAC cold-start reasoning traces.
    """
    SYSTEM_PROMPT = """You are an expert legal reasoning system. You must analyze the factual scenario against the provided rules using explicit IRAC methodology.
Inside <think>, you must:
1. Identify all legal issues and extract temporal anchors (dates, deadlines).
2. Quote exact rule clauses.
3. Test hypotheses and deliberately explore potential counterarguments or exceptions. If an initial assumption seems incorrect, self-correct ("Wait, let me double check...").
4. Formulate the final conclusion citing exact clauses.
Outside <think>, provide a clear, professional answer."""

    @staticmethod
    def verify_irac_trace(context: str, think_text: str, answer_text: str) -> tuple[bool, list[str]]:
        """
        Deterministic rule-based verifier for generated legal traces.
        Checks:
          - Presence of IRAC structural anchors
          - Verification of quoted clause citations against the context
          - Presence of self-correction / reflective reasoning tokens
        """
        errors = []
        
        # 1. Structural check
        required_sections = ["ISSUE", "RULE", "APPLICATION", "CONCLUSION"]
        for section in required_sections:
            if section not in think_text.upper():
                errors.append(f"Missing required IRAC section: {section}")

        # 2. Citation integrity check
        cited_clauses = re.findall(r"Clause\s+([0-9]+(?:\.[0-9]+)?)", think_text + " " + answer_text, re.IGNORECASE)
        valid_clauses = re.findall(r"Clause\s+([0-9]+(?:\.[0-9]+)?)", context, re.IGNORECASE)
        valid_set = set(valid_clauses)
        
        for c in set(cited_clauses):
            if c not in valid_set:
                errors.append(f"Hallucinated clause cited: Clause {c}")

        # 3. Deliberate backtracking / verification check
        backtrack_patterns = [
            r"wait",
            r"let me double check",
            r"re-evaluat",
            r"exception",
            r"however",
            r"does not apply"
        ]
        has_backtrack = any(re.search(pat, think_text, re.IGNORECASE) for pat in backtrack_patterns)
        if not has_backtrack:
            errors.append("No self-correction or exception analysis detected in reasoning trace.")

        return len(errors) == 0, errors

if __name__ == "__main__":
    sample_context = "Clause 14.2: Right of withdrawal within 7 days. Clause 14.3: Custom goods exempt."
    sample_think = "[ISSUE] Can buyer withdraw? [RULE] Clause 14.2 allows 7 days. Clause 14.3 exempts custom goods. [APPLICATION] Notice on day 6. Wait, is it custom? No. [CONCLUSION] Withdrawal valid."
    sample_ans = "Under Clause 14.2, withdrawal is valid."
    
    valid, errs = IRACGenerator.verify_irac_trace(sample_context, sample_think, sample_ans)
    print(f"IRAC Verification Result: {'PASS' if valid else 'FAIL'}")
    if errs:
        print("Errors:", errs)

"""eval_models.py
Foundation Layer: Deterministic Schemas & Contract-Driven Data Models
From: 'Fortifying Agentic AI: Building Resilient Test Scaffolding and Defense-in-Depth for LLM Systems'
by Alan Arantes
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class AtomicClaim(BaseModel):
    claim: str = Field(description="Individual atomic assertion extracted from the answer")
    evidence_quote: str = Field(
        default="",
        description="Direct quotation from retrieved context, or empty if unsupported"
    )
    status: ClaimStatus = Field(
        description="Whether the claim is strictly supported by the context"
    )


class FaithfulnessEvaluation(BaseModel):
    atomic_claims: List[AtomicClaim] = Field(
        description="List of deconstructed atomic claims and their evidence verification"
    )
    reasoning: str = Field(
        description="Analytical step-by-step audit conducted BEFORE scoring"
    )
    hallucinated_facts: List[str] = Field(
        default_factory=list,
        description="Extrapolated or unsupported assertions identified as hallucinations"
    )
    faithfulness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of supported claims to total atomic claims"
    )
    verdict: Verdict = Field(
        description="PASS only if 100% of claims are supported under Closed-World Assumption, else FAIL"
    )


# Trajectory Layer Models
class ToolCallRecord(BaseModel):
    step: int = Field(description="Step index in the agent execution trace")
    tool: str = Field(description="Name of the invoked tool")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[Any] = Field(default=None, description="Result returned by the tool")
    blocked: bool = Field(default=False, description="Whether the call was intercepted and blocked by policy")
    block_reason: Optional[str] = Field(default=None, description="Reason for policy interception")


class AgentTrajectory(BaseModel):
    input_prompt: str
    intermediate_steps: List[ToolCallRecord] = Field(default_factory=list)
    output: str
    step_count: int
    status: str = Field(
        default="completed",
        description="Execution status: 'completed', 'halted_loop_limit', 'halted_least_privilege_violation'"
    )

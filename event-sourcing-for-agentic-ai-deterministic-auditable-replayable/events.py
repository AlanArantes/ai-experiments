"""events.py
Event Sourcing for Agentic AI: Domain Event Models & Immutable Contracts
From: 'Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems'
by Alan Arantes
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    SESSION_STARTED = "SessionStarted"
    PROMPT_RECEIVED = "PromptReceived"
    COGNITION_PLANNED = "CognitionPlanned"
    TOOL_EXECUTION_REQUESTED = "ToolExecutionRequested"
    POLICY_EVALUATED = "PolicyEvaluated"
    TOOL_EXECUTION_COMPLETED = "ToolExecutionCompleted"
    OUTPUT_GENERATED = "OutputGenerated"
    SESSION_HALTED = "SessionHalted"
    CHECKPOINT_CREATED = "CheckpointCreated"


class BaseAgentEvent(BaseModel):
    """Immutable event representing a factual occurrence in an agent's execution lifecycle."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(description="Unique identifier of the agent session")
    sequence_number: int = Field(description="Strict monotonically increasing event index")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: EventType = Field(description="Domain event discriminator")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload capturing state changes, observations, or intentions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata (model, actor, latency, prev_hash)"
    )
    causation_id: Optional[str] = Field(
        default=None,
        description="ID of the event that directly caused this event"
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="ID correlating all events in a single interaction turn"
    )

    model_config = ConfigDict(frozen=True)  # Enforce immutability

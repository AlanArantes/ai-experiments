"""agent_state.py
State as a Pure Projection (Fold/Aggregate) over Event Streams
From: 'Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems'
by Alan Arantes
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from events import BaseAgentEvent, EventType


class AgentState(BaseModel):
    """Materialized projection of an agent's state at a specific point in time."""

    session_id: str
    status: str = "INITIALIZED"
    current_sequence: int = 0
    user_query: Optional[str] = None
    plan: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    working_memory: Dict[str, Any] = Field(default_factory=dict)
    tools_invoked: List[Dict[str, Any]] = Field(default_factory=list)
    final_output: Optional[str] = None
    error_message: Optional[str] = None
    policy_violations: int = 0


def apply_event(state: AgentState, event: BaseAgentEvent) -> AgentState:
    """Pure, deterministic reducer function that applies a single event to a state projection."""
    # Ensure immutability by copying state
    new_state = state.model_copy(deep=True)
    new_state.current_sequence = event.sequence_number

    match event.event_type:
        case EventType.SESSION_STARTED:
            new_state.status = "RUNNING"
            new_state.working_memory.update(event.payload.get("initial_context", {}))

        case EventType.PROMPT_RECEIVED:
            prompt = event.payload.get("prompt", "")
            new_state.user_query = prompt
            new_state.conversation_history.append({"role": "user", "content": prompt})
            new_state.status = "PROCESSING"

        case EventType.COGNITION_PLANNED:
            plan = event.payload.get("plan", "")
            new_state.plan = plan
            new_state.conversation_history.append({"role": "assistant", "thought": plan})

        case EventType.TOOL_EXECUTION_REQUESTED:
            tool_call = {
                "tool": event.payload.get("tool"),
                "args": event.payload.get("args"),
                "status": "PENDING"
            }
            new_state.tools_invoked.append(tool_call)
            new_state.status = "AWAITING_TOOL"

        case EventType.POLICY_EVALUATED:
            allowed = event.payload.get("allowed", True)
            if not allowed:
                new_state.policy_violations += 1
                if new_state.tools_invoked:
                    new_state.tools_invoked[-1]["status"] = "BLOCKED"
                    new_state.tools_invoked[-1]["block_reason"] = event.payload.get("reason")

        case EventType.TOOL_EXECUTION_COMPLETED:
            result = event.payload.get("result")
            tool_name = event.payload.get("tool")
            if new_state.tools_invoked:
                new_state.tools_invoked[-1]["status"] = "COMPLETED"
                new_state.tools_invoked[-1]["result"] = result
            # Update working memory with observation
            new_state.working_memory[f"tool_result_{tool_name}"] = result
            new_state.conversation_history.append({
                "role": "system",
                "observation": str(result)
            })
            new_state.status = "PROCESSING"

        case EventType.OUTPUT_GENERATED:
            output = event.payload.get("output", "")
            new_state.final_output = output
            new_state.conversation_history.append({"role": "assistant", "content": output})
            new_state.status = "COMPLETED"

        case EventType.SESSION_HALTED:
            reason = event.payload.get("reason", "Unknown interruption")
            new_state.status = "HALTED"
            new_state.error_message = reason

        case EventType.CHECKPOINT_CREATED:
            # Snapshot event
            pass

    return new_state


def reconstruct_state(events: List[BaseAgentEvent], initial_state: Optional[AgentState] = None) -> AgentState:
    """Folds a sequence of domain events to deterministically reconstruct state.
    
    State_t = fold(State_0, [Event_1, ..., Event_t])
    """
    if not events:
        raise ValueError("Cannot reconstruct state from empty event stream.")

    state = initial_state or AgentState(session_id=events[0].session_id)
    for event in events:
        state = apply_event(state, event)
    return state

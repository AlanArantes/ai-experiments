"""replay_engine.py
Deterministic Replay Engine, Time-Travel Debugger & What-If Branching
From: 'Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems'
by Alan Arantes
"""

from typing import Any, Callable, Dict, List, Optional
from events import BaseAgentEvent, EventType
from event_store import EventStore
from agent_state import AgentState, reconstruct_state


class ReplayEngine:
    """Provides time-travel state reconstruction, deterministic stubbing,
    and what-if scenario branching over historical event streams.
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    def time_travel_to_sequence(self, session_id: str, sequence_number: int) -> AgentState:
        """Reconstructs the agent state exactly as it existed at sequence_number."""
        events = self.event_store.get_stream(session_id, from_sequence=1, to_sequence=sequence_number)
        if not events:
            raise ValueError(f"No events found for session '{session_id}' up to sequence {sequence_number}")
        return reconstruct_state(events)

    def extract_deterministic_stubs(self, session_id: str) -> Dict[str, Any]:
        """Extracts recorded tool results from past ToolExecutionCompleted events
        to serve as deterministic fixtures during offline replay.
        """
        events = self.event_store.get_stream(session_id)
        stubs = {}
        for event in events:
            if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
                tool_name = event.payload.get("tool")
                result = event.payload.get("result")
                stubs[tool_name] = result
        return stubs

    def fork_session_at(
        self,
        source_session_id: str,
        sequence_number: int,
        new_session_id: str
    ) -> List[BaseAgentEvent]:
        """Forks a historical stream at sequence_number to create an alternate timeline
        for 'what-if' counterfactual testing.
        """
        source_events = self.event_store.get_stream(
            source_session_id, from_sequence=1, to_sequence=sequence_number
        )
        if not source_events:
            raise ValueError(f"Cannot fork empty stream from session '{source_session_id}'")

        forked_events = []
        for e in source_events:
            forked_event = e.model_copy(update={
                "session_id": new_session_id,
                "metadata": {**e.metadata, "forked_from": source_session_id}
            })
            persisted = self.event_store.append(forked_event)
            forked_events.append(persisted)

        return forked_events

    def generate_audit_trail(self, session_id: str) -> List[Dict[str, Any]]:
        """Generates a forensic audit report detailing every interaction and decision."""
        events = self.event_store.get_stream(session_id)
        report = []
        for e in events:
            report.append({
                "sequence": e.sequence_number,
                "timestamp": e.timestamp,
                "event_type": e.event_type.value,
                "causation_id": e.causation_id,
                "payload": e.payload,
                "hash": e.metadata.get("hash", "")[:12] + "...",
                "prev_hash": e.metadata.get("prev_hash", "")[:12] + "..."
            })
        return report

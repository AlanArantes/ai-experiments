"""tests/test_event_sourcing.py
Comprehensive Pytest Suite for Event Sourcing in Agentic AI Systems
From: 'Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems'
by Alan Arantes
"""

import pytest
from events import BaseAgentEvent, EventType
from event_store import EventStore, EventStoreConcurrencyError, EventStoreTamperError
from agent_state import AgentState, reconstruct_state, apply_event
from replay_engine import ReplayEngine
from agent_orchestrator import EventSourcedAgent


# ========================================================
# 1. Event Store & Cryptographic Immutability
# ========================================================
def test_event_store_sequential_append():
    """Verify strictly monotonic sequence ordering in event store."""
    store = EventStore()
    e1 = BaseAgentEvent(session_id="s1", sequence_number=1, event_type=EventType.SESSION_STARTED)
    e2 = BaseAgentEvent(session_id="s1", sequence_number=2, event_type=EventType.PROMPT_RECEIVED)

    store.append(e1)
    store.append(e2)

    stream = store.get_stream("s1")
    assert len(stream) == 2
    assert stream[0].sequence_number == 1
    assert stream[1].sequence_number == 2


def test_event_store_concurrency_conflict():
    """Verify that out-of-order sequence insertion raises EventStoreConcurrencyError."""
    store = EventStore()
    e1 = BaseAgentEvent(session_id="s1", sequence_number=1, event_type=EventType.SESSION_STARTED)
    store.append(e1)

    # Skipping sequence 2 and appending 3 must raise concurrency error
    e3 = BaseAgentEvent(session_id="s1", sequence_number=3, event_type=EventType.PROMPT_RECEIVED)
    with pytest.raises(EventStoreConcurrencyError):
        store.append(e3)


def test_cryptographic_tamper_detection():
    """Verify that tampering with an immutable event invalidates the hash chain."""
    store = EventStore()
    e1 = BaseAgentEvent(session_id="s1", sequence_number=1, event_type=EventType.SESSION_STARTED)
    e2 = BaseAgentEvent(
        session_id="s1",
        sequence_number=2,
        event_type=EventType.PROMPT_RECEIVED,
        payload={"prompt": "Transfer $100"}
    )
    store.append(e1)
    store.append(e2)

    # Verify initial integrity passes
    assert store.verify_stream_integrity("s1") is True

    # Malicious actor mutates stored event payload directly
    store._streams["s1"][1] = store._streams["s1"][1].model_copy(
        update={"payload": {"prompt": "Transfer $1,000,000"}}
    )

    # Cryptographic verification must catch the discrepancy
    with pytest.raises(EventStoreTamperError):
        store.verify_stream_integrity("s1")


# ========================================================
# 2. Pure State Projection (Fold / Aggregate)
# ========================================================
def test_deterministic_state_projection():
    """Verify that state is a pure deterministic projection over events."""
    events = [
        BaseAgentEvent(session_id="s1", sequence_number=1, event_type=EventType.SESSION_STARTED, payload={"initial_context": {"role": "analyst"}}),
        BaseAgentEvent(session_id="s1", sequence_number=2, event_type=EventType.PROMPT_RECEIVED, payload={"prompt": "Check status"}),
        BaseAgentEvent(session_id="s1", sequence_number=3, event_type=EventType.OUTPUT_GENERATED, payload={"output": "All systems operational"}),
    ]

    # Reconstruct twice: both must be identical
    state_a = reconstruct_state(events)
    state_b = reconstruct_state(events)

    assert state_a.session_id == "s1"
    assert state_a.status == "COMPLETED"
    assert state_a.user_query == "Check status"
    assert state_a.final_output == "All systems operational"
    assert state_a.model_dump() == state_b.model_dump()


# ========================================================
# 3. Agent Orchestrator & Policy Enforcement
# ========================================================
def test_agent_orchestrator_execution_flow():
    """Verify end-to-end agent execution emits proper domain events."""
    store = EventStore()
    agent = EventSourcedAgent(session_id="sess-100", event_store=store)

    state = agent.run("Show details for order ORD-101", is_read_only=True)

    stream = store.get_stream("sess-100")
    event_types = [e.event_type for e in stream]

    assert event_types == [
        EventType.SESSION_STARTED,
        EventType.PROMPT_RECEIVED,
        EventType.COGNITION_PLANNED,
        EventType.TOOL_EXECUTION_REQUESTED,
        EventType.POLICY_EVALUATED,
        EventType.TOOL_EXECUTION_COMPLETED,
        EventType.OUTPUT_GENERATED,
        EventType.CHECKPOINT_CREATED,
    ]
    assert state.status == "COMPLETED"
    assert "Cloud Server 8-Core" in str(state.final_output)


def test_agent_orchestrator_policy_interception():
    """Verify least privilege policy blocks destructive financial writes in read-only mode."""
    store = EventStore()
    agent = EventSourcedAgent(session_id="sess-200", event_store=store)

    state = agent.run("Process refund for ORD-101", is_read_only=True)

    stream = store.get_stream("sess-200")
    event_types = [e.event_type for e in stream]

    assert EventType.POLICY_EVALUATED in event_types
    assert EventType.SESSION_HALTED in event_types
    assert EventType.TOOL_EXECUTION_COMPLETED not in event_types  # Tool must NEVER have run
    assert state.status == "HALTED"
    assert state.policy_violations == 1


# ========================================================
# 4. Replay Engine: Time-Travel & Branching
# ========================================================
def test_time_travel_debugging():
    """Verify reconstructing state at past sequence points."""
    store = EventStore()
    agent = EventSourcedAgent(session_id="sess-300", event_store=store)
    agent.run("Show details for order ORD-101", is_read_only=True)

    replay = ReplayEngine(store)

    # Reconstruct state at sequence 2 (Prompt Received)
    past_state_seq2 = replay.time_travel_to_sequence("sess-300", sequence_number=2)
    assert past_state_seq2.current_sequence == 2
    assert past_state_seq2.status == "PROCESSING"
    assert past_state_seq2.final_output is None

    # Reconstruct state at sequence 6 (Tool Completed)
    past_state_seq6 = replay.time_travel_to_sequence("sess-300", sequence_number=6)
    assert past_state_seq6.current_sequence == 6
    assert len(past_state_seq6.tools_invoked) == 1
    assert past_state_seq6.tools_invoked[0]["status"] == "COMPLETED"


def test_what_if_branching():
    """Verify forking an execution stream to explore counterfactual scenario."""
    store = EventStore()
    agent = EventSourcedAgent(session_id="sess-original", event_store=store)
    agent.run("Calculate refund for ORD-101", is_read_only=True)

    replay = ReplayEngine(store)
    forked_events = replay.fork_session_at(
        source_session_id="sess-original",
        sequence_number=2,  # Fork right after PromptReceived
        new_session_id="sess-forked"
    )

    assert len(forked_events) == 2
    forked_stream = store.get_stream("sess-forked")
    assert len(forked_stream) == 2
    assert forked_stream[0].session_id == "sess-forked"
    assert forked_stream[0].metadata.get("forked_from") == "sess-original"

    # Run alternate agent on forked stream
    forked_agent = EventSourcedAgent(session_id="sess-forked", event_store=store)
    forked_state = forked_agent.run("Show details for order ORD-101", is_read_only=True)
    assert forked_state.status == "COMPLETED"
    assert store.verify_stream_integrity("sess-forked") is True

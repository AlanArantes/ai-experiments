"""test_runner.py
Interactive CLI Demonstration of Event Sourcing for Agentic AI
From: 'Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems'
by Alan Arantes
"""

import json
from events import EventType
from event_store import EventStore, EventStoreTamperError
from agent_state import reconstruct_state
from replay_engine import ReplayEngine
from agent_orchestrator import EventSourcedAgent


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_event_sourced_execution():
    print_banner("1. AGENT INTERACTION: EMITTING IMMUTABLE DOMAIN EVENTS")
    store = EventStore()
    agent = EventSourcedAgent(session_id="session-finance-01", event_store=store)

    state = agent.run("Show details for order ORD-101", is_read_only=True)
    stream = store.get_stream("session-finance-01")

    print(f"Total events emitted: {len(stream)}")
    print(f"Final Projected State Status: {state.status}")
    print(f"Final Output: {state.final_output}")

    print("\n[Event Stream Timeline]")
    for e in stream:
        print(f"  Seq #{e.sequence_number:02d} | [{e.event_type.value:<24}] | Hash: {e.metadata.get('hash', '')[:8]}...")


def demo_deterministic_projection():
    print_banner("2. STATE AS A PURE PROJECTION (STATE = FOLD(STATE_0, EVENTS))")
    store = EventStore()
    agent = EventSourcedAgent(session_id="session-fold-test", event_store=store)
    agent.run("Calculate refund for ORD-101", is_read_only=True)

    stream = store.get_stream("session-fold-test")
    
    # Run fold twice on the same stream
    state_pass1 = reconstruct_state(stream)
    state_pass2 = reconstruct_state(stream)

    assert state_pass1.model_dump() == state_pass2.model_dump()
    print("[PASS] Deterministic Fold Invariant verified:")
    print(f"  - Reconstructed Status: {state_pass1.status}")
    print(f"  - Working Memory Keys: {list(state_pass1.working_memory.keys())}")
    print(f"  - Invariant: State(Run 1) == State(Run 2) is TRUE")


def demo_time_travel_debugging():
    print_banner("3. TIME-TRAVEL DEBUGGING: INSPECTING STATE ACROSS TIME")
    store = EventStore()
    agent = EventSourcedAgent(session_id="session-timetravel", event_store=store)
    agent.run("Show details for order ORD-101", is_read_only=True)

    replay = ReplayEngine(store)

    # Time travel to sequence 2 (Prompt Received)
    state_seq2 = replay.time_travel_to_sequence("session-timetravel", 2)
    print(f"[Time-Travel @ Sequence 2 (Prompt Received)]")
    print(f"  Status: {state_seq2.status} | Final Output: {state_seq2.final_output} | Tools: {len(state_seq2.tools_invoked)}")

    # Time travel to sequence 6 (Tool Completed)
    state_seq6 = replay.time_travel_to_sequence("session-timetravel", 6)
    print(f"\n[Time-Travel @ Sequence 6 (Tool Observation)]")
    print(f"  Status: {state_seq6.status} | Tools Invoked: {state_seq6.tools_invoked[0]['tool']} -> Status: {state_seq6.tools_invoked[0]['status']}")

    # Time travel to sequence 7 (Output Emitted)
    state_seq7 = replay.time_travel_to_sequence("session-timetravel", 7)
    print(f"\n[Time-Travel @ Sequence 7 (Final Response)]")
    print(f"  Status: {state_seq7.status} | Output: {state_seq7.final_output}")


def demo_policy_enforcement():
    print_banner("4. LEAST PRIVILEGE POLICY AUDITING")
    store = EventStore()
    agent = EventSourcedAgent(session_id="session-security-audit", event_store=store)

    # Attempt financial write during read-only interaction
    state = agent.run("Process refund for ORD-101", is_read_only=True)
    stream = store.get_stream("session-security-audit")

    print(f"Session Status: {state.status}")
    print(f"Security Interceptions: {state.policy_violations}")
    print(f"Error Message: {state.error_message}")
    
    blocked_event = [e for e in stream if e.event_type == EventType.POLICY_EVALUATED][0]
    print(f"Policy Event Payload: {json.dumps(blocked_event.payload, indent=2)}")


def demo_tamper_detection():
    print_banner("5. CRYPTOGRAPHIC PROVENANCE & TAMPER DETECTION")
    store = EventStore()
    agent = EventSourcedAgent(session_id="session-tamper-proof", event_store=store)
    agent.run("Show details for order ORD-101", is_read_only=True)

    print("[PASS] Verifying uncorrupted event stream integrity...")
    assert store.verify_stream_integrity("session-tamper-proof") is True
    print("  [OK] Cryptographic hash chain verified successfully.")

    print("\n[Simulating Malicious Historical Tampering]")
    # Tamper with event sequence 2
    store._streams["session-tamper-proof"][1] = store._streams["session-tamper-proof"][1].model_copy(
        update={"payload": {"prompt": "MALICIOUS OVERWRITE: DROP DATABASE;"}}
    )
    try:
        store.verify_stream_integrity("session-tamper-proof")
    except EventStoreTamperError as e:
        print(f"  [OK] Tamper Detected: {e}")


def demo_what_if_branching():
    print_banner("6. COUNTERFACTUAL 'WHAT-IF' BRANCHING")
    store = EventStore()
    agent = EventSourcedAgent(session_id="main-timeline", event_store=store)
    agent.run("Calculate refund for ORD-101", is_read_only=True)

    replay = ReplayEngine(store)
    print("Forking 'main-timeline' at sequence 2 into 'alternate-timeline'...")
    replay.fork_session_at("main-timeline", sequence_number=2, new_session_id="alternate-timeline")

    forked_agent = EventSourcedAgent(session_id="alternate-timeline", event_store=store)
    forked_state = forked_agent.run("Show details for order ORD-101", is_read_only=True)

    print(f"Original Timeline Event Count: {len(store.get_stream('main-timeline'))}")
    print(f"Forked Timeline Event Count: {len(store.get_stream('alternate-timeline'))}")
    print(f"Forked State Output: {forked_state.final_output}")


def main():
    demo_event_sourced_execution()
    demo_deterministic_projection()
    demo_time_travel_debugging()
    demo_policy_enforcement()
    demo_tamper_detection()
    demo_what_if_branching()
    print("\n[SUCCESS] All Event Sourcing capabilities demonstrated successfully!")


if __name__ == "__main__":
    main()

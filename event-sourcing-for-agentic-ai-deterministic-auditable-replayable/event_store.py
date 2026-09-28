"""event_store.py
Append-Only Event Store with Hash-Chained Tamper Detection
From: 'Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems'
by Alan Arantes
"""

import hashlib
import json
from typing import Dict, List, Optional
from events import BaseAgentEvent


class EventStoreConcurrencyError(Exception):
    """Raised when an event does not match the expected sequential monotonic order."""
    pass


class EventStoreTamperError(Exception):
    """Raised when cryptographic verification detects historical event mutation."""
    pass


class EventStore:
    """Immutable, append-only event store providing durable persistence,
    strict monotonic sequencing, and hash-chained cryptographic provenance.
    """

    def __init__(self):
        # In-memory stream partition: session_id -> List[BaseAgentEvent]
        self._streams: Dict[str, List[BaseAgentEvent]] = {}

    def _compute_hash(self, prev_hash: str, event: BaseAgentEvent) -> str:
        """Computes SHA256 hash connecting this event to its predecessor."""
        hasher = hashlib.sha256()
        payload_str = json.dumps(event.payload, sort_keys=True)
        token = f"{prev_hash}|{event.session_id}|{event.sequence_number}|{event.event_type.value}|{payload_str}"
        hasher.update(token.encode("utf-8"))
        return hasher.hexdigest()

    def append(self, event: BaseAgentEvent) -> BaseAgentEvent:
        """Appends a new event to the stream, enforcing strict monotonic ordering
        and updating the cryptographic hash chain.
        """
        session_id = event.session_id
        if session_id not in self._streams:
            self._streams[session_id] = []

        stream = self._streams[session_id]
        expected_seq = len(stream) + 1

        if event.sequence_number != expected_seq:
            raise EventStoreConcurrencyError(
                f"Concurrency conflict for session '{session_id}': "
                f"expected sequence {expected_seq}, but received {event.sequence_number}"
            )

        prev_hash = stream[-1].metadata.get("hash", "GENESIS") if stream else "GENESIS"
        current_hash = self._compute_hash(prev_hash, event)

        # Create updated event with hash chain metadata
        updated_metadata = dict(event.metadata)
        updated_metadata["prev_hash"] = prev_hash
        updated_metadata["hash"] = current_hash

        persisted_event = event.model_copy(update={"metadata": updated_metadata})
        stream.append(persisted_event)
        return persisted_event

    def get_stream(
        self,
        session_id: str,
        from_sequence: int = 1,
        to_sequence: Optional[int] = None
    ) -> List[BaseAgentEvent]:
        """Retrieves a slice of events from a given session stream."""
        if session_id not in self._streams:
            return []

        stream = self._streams[session_id]
        filtered = [
            e for e in stream
            if e.sequence_number >= from_sequence and (to_sequence is None or e.sequence_number <= to_sequence)
        ]
        return filtered

    def verify_stream_integrity(self, session_id: str) -> bool:
        """Verifies that the event stream has not been tampered with or corrupted."""
        stream = self.get_stream(session_id)
        if not stream:
            return True

        prev_hash = "GENESIS"
        for event in stream:
            expected_hash = self._compute_hash(prev_hash, event)
            stored_hash = event.metadata.get("hash")
            if stored_hash != expected_hash:
                raise EventStoreTamperError(
                    f"Integrity check failed at sequence {event.sequence_number}: "
                    f"stored hash '{stored_hash}' != computed hash '{expected_hash}'"
                )
            prev_hash = expected_hash

        return True

    def list_sessions(self) -> List[str]:
        return list(self._streams.keys())

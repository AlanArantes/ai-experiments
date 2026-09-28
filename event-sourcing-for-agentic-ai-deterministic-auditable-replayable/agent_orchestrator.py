"""agent_orchestrator.py
Event-Sourced Agent Orchestrator: Emitting Domain Events & Projecting State
From: 'Event Sourcing for Agentic AI: Architecting Deterministic, Auditable, and Replayable LLM Systems'
by Alan Arantes
"""

import os
from typing import Any, Callable, Dict, Optional
from events import BaseAgentEvent, EventType
from event_store import EventStore
from agent_state import AgentState, reconstruct_state


class MockEnterpriseTools:
    """Enterprise backend tools executed by the agent."""

    @staticmethod
    def get_order_details(order_id: str) -> Dict[str, Any]:
        orders = {
            "ORD-101": {"order_id": "ORD-101", "item": "Cloud Server 8-Core", "amount": 250.00, "status": "Shipped"},
            "ORD-102": {"order_id": "ORD-102", "item": "Database Storage 1TB", "amount": 120.00, "status": "Processing"}
        }
        return orders.get(order_id, {"error": f"Order '{order_id}' not found"})

    @staticmethod
    def calculate_refund_amount(order_id: str, reason: str) -> Dict[str, Any]:
        return {"order_id": order_id, "refund_eligible": True, "eligible_amount": 250.00, "reason": reason}

    @staticmethod
    def process_refund(order_id: str, amount: float) -> Dict[str, Any]:
        return {"order_id": order_id, "amount_refunded": amount, "transaction_id": "TX-998877", "status": "COMPLETED"}


class EventSourcedAgent:
    """Agentic AI Orchestrator that strictly records every decision,
    action, and observation as an immutable event in the EventStore.
    """

    def __init__(
        self,
        session_id: str,
        event_store: EventStore,
        tools: Optional[Dict[str, Callable]] = None,
        deterministic_stubs: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.event_store = event_store
        self.deterministic_stubs = deterministic_stubs or {}
        self.tools = tools or {
            "get_order_details": MockEnterpriseTools.get_order_details,
            "calculate_refund_amount": MockEnterpriseTools.calculate_refund_amount,
            "process_refund": MockEnterpriseTools.process_refund
        }

    def _next_sequence(self) -> int:
        stream = self.event_store.get_stream(self.session_id)
        return len(stream) + 1

    def _emit(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        causation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaseAgentEvent:
        event = BaseAgentEvent(
            session_id=self.session_id,
            sequence_number=self._next_sequence(),
            event_type=event_type,
            payload=payload,
            causation_id=causation_id,
            metadata=metadata or {}
        )
        return self.event_store.append(event)

    def get_current_state(self) -> AgentState:
        stream = self.event_store.get_stream(self.session_id)
        if not stream:
            return AgentState(session_id=self.session_id)
        return reconstruct_state(stream)

    def run(
        self,
        prompt: str,
        role: str = "customer_support",
        is_read_only: bool = True
    ) -> AgentState:
        """Executes a cognitive interaction turn with complete event-sourcing."""
        stream = self.event_store.get_stream(self.session_id)
        
        # 1. Initialize session if not started
        last_event_id = None
        if not stream:
            e_start = self._emit(
                EventType.SESSION_STARTED,
                payload={"role": role, "initial_context": {"tenant": "AcmeEnterprise"}}
            )
            last_event_id = e_start.event_id

        # 2. Prompt Received
        e_prompt = self._emit(
            EventType.PROMPT_RECEIVED,
            payload={"prompt": prompt, "role": role},
            causation_id=last_event_id
        )
        last_event_id = e_prompt.event_id

        # 3. Cognition / Planning (Deterministic inference based on prompt)
        p_lower = prompt.lower()
        if "order" in p_lower and "details" in p_lower:
            plan = "Lookup customer order ORD-101 in database"
            tool_call = {"tool": "get_order_details", "args": {"order_id": "ORD-101"}}
        elif "refund" in p_lower and "process" in p_lower:
            plan = "Issue full financial refund for order ORD-101"
            tool_call = {"tool": "process_refund", "args": {"order_id": "ORD-101", "amount": 250.00}}
        elif "refund" in p_lower:
            plan = "Calculate refund eligibility for order ORD-101"
            tool_call = {"tool": "calculate_refund_amount", "args": {"order_id": "ORD-101", "reason": "damaged"}}
        else:
            plan = "Answer customer query with policy information"
            tool_call = None

        e_cog = self._emit(
            EventType.COGNITION_PLANNED,
            payload={"plan": plan, "selected_tool": tool_call.get("tool") if tool_call else None},
            causation_id=last_event_id
        )
        last_event_id = e_cog.event_id

        # 4. Tool Execution / Policy Check
        if tool_call:
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            e_req = self._emit(
                EventType.TOOL_EXECUTION_REQUESTED,
                payload={"tool": tool_name, "args": tool_args},
                causation_id=last_event_id
            )
            last_event_id = e_req.event_id

            # Policy Check: Block write operations if context is read-only
            if is_read_only and tool_name == "process_refund":
                e_pol = self._emit(
                    EventType.POLICY_EVALUATED,
                    payload={"allowed": False, "reason": "Least Privilege: Financial write blocked in read-only mode"},
                    causation_id=last_event_id
                )
                self._emit(
                    EventType.SESSION_HALTED,
                    payload={"reason": "Security policy violation: unauthorized financial refund attempt"},
                    causation_id=e_pol.event_id
                )
                return self.get_current_state()

            # Allowed Policy
            e_pol = self._emit(
                EventType.POLICY_EVALUATED,
                payload={"allowed": True, "reason": "Authorized read inquiry"},
                causation_id=last_event_id
            )
            last_event_id = e_pol.event_id

            # Execute tool or retrieve deterministic stub
            if tool_name in self.deterministic_stubs:
                result = self.deterministic_stubs[tool_name]
            elif tool_name in self.tools:
                result = self.tools[tool_name](**tool_args)
            else:
                result = {"error": f"Tool '{tool_name}' not available"}

            e_res = self._emit(
                EventType.TOOL_EXECUTION_COMPLETED,
                payload={"tool": tool_name, "args": tool_args, "result": result},
                causation_id=last_event_id
            )
            last_event_id = e_res.event_id
            final_text = f"Information retrieved for {tool_name}: {result}"
        else:
            final_text = "I have processed your request based on standard operating procedures."

        # 5. Output Generated
        e_out = self._emit(
            EventType.OUTPUT_GENERATED,
            payload={"output": final_text},
            causation_id=last_event_id
        )

        # 6. Checkpoint Created
        self._emit(
            EventType.CHECKPOINT_CREATED,
            payload={"checkpoint_type": "post_turn"},
            causation_id=e_out.event_id
        )

        return self.get_current_state()

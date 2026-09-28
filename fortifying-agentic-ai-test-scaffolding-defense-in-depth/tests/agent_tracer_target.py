"""tests/agent_tracer_target.py
Custom Target Bridge for Promptfoo Trajectory Telemetry and Assertion Running
From: 'Fortifying Agentic AI: Building Resilient Test Scaffolding and Defense-in-Depth for LLM Systems'
by Alan Arantes
"""

import os
from typing import Any, Dict, List, Optional
import requests

# Try importing local runtime as direct fallback if agent service is not running via HTTP
try:
    from agent_runtime import AgentRuntimeHarness
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from agent_runtime import AgentRuntimeHarness


def call_api(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Promptfoo custom provider bridge function.
    
    Routes test execution to the agent endpoint, captures intermediate execution
    trajectories, and returns output along with audit telemetry.
    """
    api_url = os.getenv("AGENT_ENDPOINT", "http://localhost:8000/api/agent/run")
    api_key = os.getenv("API_KEY", "test-key-123")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Attempt HTTP invocation if backend server is available
    try:
        response = requests.post(
            api_url,
            json={"input": prompt, "enable_tracing": True},
            headers=headers,
            timeout=2.0
        )
        if response.status_code == 200:
            data = response.json()
            steps = data.get("intermediate_steps", [])
            return {
                "output": data.get("output", ""),
                "metadata": {
                    "intermediate_steps": steps,
                    "step_count": data.get("step_count", len(steps)),
                    "status": data.get("status", "completed")
                }
            }
    except Exception:
        # Fallback to local in-process AgentRuntimeHarness
        pass

    # Direct in-process execution fallback
    harness = AgentRuntimeHarness(max_steps=5, enforce_least_privilege=True)
    is_read_only = True
    if context and "vars" in context and context["vars"].get("role") == "admin":
        is_read_only = False

    trajectory = harness.execute_plan(prompt=prompt, context_is_read_only=is_read_only)
    
    return {
        "output": trajectory.output,
        "metadata": {
            "intermediate_steps": [s.model_dump() for s in trajectory.intermediate_steps],
            "step_count": trajectory.step_count,
            "status": trajectory.status
        }
    }


if __name__ == "__main__":
    # Test execution bridge directly
    res = call_api("Can I cancel my order ORD-99?")
    print("Call API result:", res)

"""
Deterministic Python-based Verifier for Kubernetes SRE RLVR.
Executes pod lifecycle invariant checks and service probe requests.
"""

import subprocess
import requests
import json
import time
import sys

def verify_cluster_state(kubeconfig: str, namespace: str = "sandbox-app", service: str = "payment-processor", probe_url: str = None) -> tuple[bool, str]:
    # 1. Pod status check via kubectl
    try:
        cmd = [
            "kubectl", "--kubeconfig", kubeconfig,
            "get", "pods", "-n", namespace,
            "-o", "json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            return False, f"kubectl get pods failed: {res.stderr}"
        
        pod_data = json.loads(res.stdout)
        items = pod_data.get("items", [])
        if not items:
            return False, "No pods found in namespace"

        for pod in items:
            pod_name = pod["metadata"]["name"]
            statuses = pod.get("status", {}).get("containerStatuses", [])
            for c in statuses:
                waiting = c.get("state", {}).get("waiting", {})
                reason = waiting.get("reason", "")
                if reason in ["CrashLoopBackOff", "CreateContainerConfigError", "ImagePullBackOff", "ErrImagePull"]:
                    return False, f"Pod {pod_name} is failing with {reason}"
                if not c.get("ready", False):
                    # Pod container not ready
                    pass
    except Exception as e:
        return False, f"Error inspecting pods: {str(e)}"

    # 2. HTTP Probe check
    if probe_url:
        try:
            for _ in range(3):
                resp = requests.get(probe_url, timeout=3)
                if resp.status_code != 200:
                    return False, f"Probe returned HTTP {resp.status_code}"
                time.sleep(0.5)
        except Exception as e:
            return False, f"Probe request failed: {str(e)}"

    return True, "Cluster healthy and verified"

if __name__ == "__main__":
    kubeconfig = sys.argv[1] if len(sys.argv) > 1 else ""
    healthy, msg = verify_cluster_state(kubeconfig)
    print(f"Status: {'PASS' if healthy else 'FAIL'} - {msg}")
    sys.exit(0 if healthy else 1)

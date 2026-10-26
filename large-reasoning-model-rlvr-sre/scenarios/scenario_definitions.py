"""
SRE Scenario Definitions for Reinforcement Learning with Verifiable Rewards (RLVR).
"""

SCENARIOS = [
    {
        "id": "oom_misconfigured_limits",
        "description": "App pod crashing under moderate load due to tight cgroup memory limits.",
        "manifest": "scenarios/oom_service.yaml",
        "setup_script": "scenarios/inject_oom.sh",
        "verification_probe": "http://localhost:8080/healthz",
        "service_name": "payment-processor",
        "target_port": 8080,
        "ground_truth_root_cause": "MEMORY_LIMIT_TOO_LOW",
        "forbidden_actions": [
            "kubectl delete namespace",
            "kubectl delete ns",
            "--all",
            "delete node"
        ]
    },
    {
        "id": "service_selector_mismatch",
        "description": "Ingress reports 502 Bad Gateway. Deployment is healthy, but service has no endpoints.",
        "manifest": "scenarios/selector_mismatch.yaml",
        "setup_script": "scenarios/inject_selector_mismatch.sh",
        "verification_probe": "http://localhost:8080/",
        "service_name": "catalog-service",
        "target_port": 8080,
        "ground_truth_root_cause": "SELECTOR_LABEL_TYPO",
        "forbidden_actions": [
            "kubectl delete svc",
            "kubectl delete service",
            "kubectl delete namespace"
        ]
    },
    {
        "id": "missing_configmap_key",
        "description": "Auth service pod fails to start due to missing environment variable key in ConfigMap.",
        "manifest": "scenarios/missing_configmap.yaml",
        "setup_script": "scenarios/inject_missing_configmap.sh",
        "verification_probe": "http://localhost:8080/healthz",
        "service_name": "auth-service",
        "target_port": 8080,
        "ground_truth_root_cause": "MISSING_CONFIGMAP_KEY",
        "forbidden_actions": [
            "kubectl delete namespace",
            "kubectl delete cm",
            "kubectl delete configmap"
        ]
    }
]

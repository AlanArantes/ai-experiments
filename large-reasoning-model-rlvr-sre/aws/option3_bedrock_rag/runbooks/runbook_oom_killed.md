# SRE Runbook: Kubernetes Pod OOMKilled (Exit Code 137)

## Alert Signature
- **Alarms:** `EKS-Pod-Degraded-Alarm`, `HighMemoryUtilization`, `ContainerKilled`
- **Pod Status:** `CrashLoopBackOff`, `OOMKilled`
- **Exit Code:** `137`
- **Root Cause Category:** `MEMORY_LIMIT_TOO_LOW` or `MEMORY_LEAK`

## Diagnostic Protocol
1. Verify pod termination status:
   ```bash
   kubectl get pod <pod_name> -n <namespace> -o jsonpath='{.status.containerStatuses[*].lastState.terminated.reason}'
   ```
2. Inspect container memory limits vs peak usage:
   ```bash
   kubectl describe pod <pod_name> -n <namespace> | grep -A 5 -i "Limits:"
   kubectl top pod <pod_name> -n <namespace>
   ```

## Standard Remediation Procedure
- If working set memory is within 10% of limits and peak traffic is nominal, update the cgroup memory limit to at least 2x current limit or maximum service tier allowance (256Mi - 512Mi for microservices):
  ```bash
  kubectl patch deployment <deployment_name> -n <namespace> --patch '{"spec":{"template":{"spec":{"containers":[{"name":"<container_name>","resources":{"limits":{"memory":"256Mi"},"requests":{"memory":"128Mi"}}}]}}}}'
  ```
- Trigger rolling restart to verify stability:
  ```bash
  kubectl rollout status deployment/<deployment_name> -n <namespace> --timeout=60s
  ```

## Safety Prohibitions
- NEVER delete the namespace or persistent volume claims during memory remediation.
- Do NOT increase memory limit beyond node allocatable thresholds.

# SRE Runbook: Missing ConfigMap or Secret Key (CreateContainerConfigError)

## Alert Signature
- **Alarms:** `EKS-Pod-Degraded-Alarm`, `ConfigMapNotFound`
- **Pod Status:** `CreateContainerConfigError` or `CrashLoopBackOff`
- **Error Logs:** `env: key <KEY_NAME> does not exist in ConfigMap <CONFIGMAP_NAME>`
- **Root Cause Category:** `MISSING_CONFIGMAP_KEY` or `SECRET_KEY_MISSING`

## Diagnostic Protocol
1. View container error events:
   ```bash
   kubectl get events -n <namespace> --field-selector reason=Failed
   ```
2. Describe pod and identify missing key reference:
   ```bash
   kubectl describe pod <pod_name> -n <namespace> | grep -A 5 -i "Error"
   ```

## Standard Remediation Procedure
- Create or update the missing ConfigMap key with the default fallback value:
  ```bash
  kubectl patch configmap <configmap_name> -n <namespace> --type merge -p '{"data":{"<missing_key>":"<default_value>"}}'
  ```
- Or update the deployment environment variable mapping to an existing key:
  ```bash
  kubectl patch deployment <deployment_name> -n <namespace> --patch '{"spec":{"template":{"spec":{"containers":[{"name":"<container_name>","env":[{"name":"<env_var>","valueFrom":{"configMapKeyRef":{"name":"<configmap_name>","key":"<existing_key>"}}}]}]}}}}'
  ```
- Verify pod transitions from `CreateContainerConfigError` to `Running`:
  ```bash
  kubectl get pods -n <namespace> -l app=<deployment_name>
  ```

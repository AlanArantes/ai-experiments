# SRE Runbook: Ingress 502 Bad Gateway / Service Selector Mismatch

## Alert Signature
- **Alarms:** `ALB-502-Rate-High`, `IngressTargetUnhealthy`
- **HTTP Status:** 502 Bad Gateway
- **Pod Status:** Running / Healthy, but Service Endpoints count is 0
- **Root Cause Category:** `SELECTOR_LABEL_TYPO` or `TARGET_PORT_MISMATCH`

## Diagnostic Protocol
1. Check service endpoints:
   ```bash
   kubectl get endpoints <service_name> -n <namespace>
   ```
2. If endpoints are `<none>`, inspect service selector labels:
   ```bash
   kubectl get svc <service_name> -n <namespace> -o jsonpath='{.spec.selector}'
   ```
3. Compare service selector labels with pod labels on deployment:
   ```bash
   kubectl get deployment <deployment_name> -n <namespace> -o jsonpath='{.spec.template.metadata.labels}'
   ```

## Standard Remediation Procedure
- If selector label has a typo (e.g. `app: paymnt` instead of `app: payment`), patch the Service selector:
  ```bash
  kubectl patch svc <service_name> -n <namespace> --patch '{"spec":{"selector":{"app":"<correct_app_label>"}}}'
  ```
- Verify endpoints populate immediately:
  ```bash
  kubectl get endpoints <service_name> -n <namespace>
  ```

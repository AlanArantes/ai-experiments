#!/usr/bin/env bash
# verifier.sh <kubeconfig> <namespace> <service_name> <target_port>
set -euo pipefail

KUBECONFIG_PATH="${1:-$HOME/.kube/config}"
NAMESPACE="${2:-sandbox-app}"
SERVICE="${3:-payment-processor}"
PORT="${4:-8080}"

echo "[*] Verifying cluster invariants for service '${SERVICE}' in '${NAMESPACE}'..."

# 1. Check Pod Lifecycle Invariants
CRASHING=$(kubectl --kubeconfig="${KUBECONFIG_PATH}" get pods -n "${NAMESPACE}" \
  -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' 2>/dev/null || true)

for reason in ${CRASHING}; do
  if [[ "${reason}" =~ (CrashLoopBackOff|CreateContainerConfigError|ImagePullBackOff) ]]; then
    echo "VERIFIER_FAIL: Pod waiting with error ${reason}"
    exit 1
  fi
done

# 2. Ephemeral Probe to verify network-level HTTP 200
PROBE_RESULT=$(kubectl --kubeconfig="${KUBECONFIG_PATH}" run verifier-probe \
  --image=curlimages/curl --restart=Never --rm -i -n "${NAMESPACE}" \
  --command -- curl -s -o /dev/null -w "%{http_code}" "http://${SERVICE}.${NAMESPACE}.svc.cluster.local:${PORT}/healthz" \
  2>/dev/null || echo "FAILED")

if [ "${PROBE_RESULT}" == "200" ]; then
  echo "VERIFIER_PASS: Service healthy and answering on probe endpoint"
  exit 0
else
  echo "VERIFIER_FAIL: Endpoint returned ${PROBE_RESULT}"
  exit 1
fi

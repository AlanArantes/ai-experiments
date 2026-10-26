#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG_PATH="${1:-$HOME/.kube/config}"
NAMESPACE="sandbox-app"

echo "[*] Applying Scenario 3: Missing ConfigMap Key..."
kubectl --kubeconfig="${KUBECONFIG_PATH}" apply -f scenarios/missing_configmap.yaml

sleep 6
echo "[*] Checking Pod Event Stream:"
kubectl --kubeconfig="${KUBECONFIG_PATH}" get events -n "${NAMESPACE}" \
  --field-selector reason=FailedMount,reason=Failed \
  --sort-by='.metadata.creationTimestamp' | tail -n 5

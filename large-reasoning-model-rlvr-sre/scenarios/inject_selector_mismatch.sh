#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG_PATH="${1:-$HOME/.kube/config}"
NAMESPACE="sandbox-app"

echo "[*] Applying Scenario 2: Service Selector Mismatch..."
kubectl --kubeconfig="${KUBECONFIG_PATH}" apply -f scenarios/selector_mismatch.yaml

echo "[*] Waiting for deployment rollout..."
kubectl --kubeconfig="${KUBECONFIG_PATH}" rollout status deployment/catalog-api -n "${NAMESPACE}" --timeout=45s

echo "[*] Verifying endpoint availability (expected: 0 endpoints)..."
ENDPOINTS=$(kubectl --kubeconfig="${KUBECONFIG_PATH}" get endpoints catalog-service -n "${NAMESPACE}" -o jsonpath='{.subsets}')
if [ -z "${ENDPOINTS}" ] || [ "${ENDPOINTS}" == "null" ]; then
    echo "[!] Verification confirmed: catalog-service has no matching pod endpoints."
fi

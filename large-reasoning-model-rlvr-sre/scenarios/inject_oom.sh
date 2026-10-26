#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG_PATH="${1:-$HOME/.kube/config}"
NAMESPACE="sandbox-app"

echo "[*] Applying Scenario 1: Memory Limit Starvation..."
kubectl --kubeconfig="${KUBECONFIG_PATH}" apply -f scenarios/oom_service.yaml

echo "[*] Waiting for initial pod deployment..."
kubectl --kubeconfig="${KUBECONFIG_PATH}" rollout status deployment/payment-processor -n "${NAMESPACE}" --timeout=60s

echo "[*] Injecting traffic burst to trigger OOMKilled..."
POD_NAME=$(kubectl --kubeconfig="${KUBECONFIG_PATH}" get pods -n "${NAMESPACE}" -l app=payment-processor -o jsonpath='{.items[0].metadata.name}')

# Execute requests inside the cluster network to force allocation burst
kubectl --kubeconfig="${KUBECONFIG_PATH}" run oom-trigger --image=curlimages/curl --restart=Never -n "${NAMESPACE}" -- \
  sh -c "for i in \$(seq 1 3); do curl -s http://payment-processor:8080/process || true; sleep 1; done"

sleep 5
echo "[*] Fault injected. Pod status check:"
kubectl --kubeconfig="${KUBECONFIG_PATH}" get pods -n "${NAMESPACE}" -l app=payment-processor

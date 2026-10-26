#!/usr/bin/env bash
# run_k3s_sandbox.sh
# Spins up an ephemeral single-node K3s cluster using Podman
set -euo pipefail

CLUSTER_NAME="${1:-k8s-sandbox-01}"
API_PORT="${2:-6443}"
OUTPUT_DIR="${3:-$(pwd)/k8s-output}"

mkdir -p "${OUTPUT_DIR}"

echo "[*] Removing existing container if present..."
podman rm -f "${CLUSTER_NAME}" 2>/dev/null || true

echo "[*] Launching ephemeral K3s cluster '${CLUSTER_NAME}' via Podman..."
podman run -d --privileged \
  --name "${CLUSTER_NAME}" \
  -p "${API_PORT}:6443" \
  -e K3S_KUBECONFIG_OUTPUT=/output/kubeconfig.yaml \
  -e K3S_KUBECONFIG_MODE="666" \
  -v "${OUTPUT_DIR}:/output:Z" \
  rancher/k3s:v1.31.0-k3s1 server --disable traefik --disable servicelb

echo "[*] Waiting for kubeconfig to be generated in ${OUTPUT_DIR}..."
KUBECONFIG_PATH="${OUTPUT_DIR}/kubeconfig.yaml"
RETRIES=30
until [ -f "${KUBECONFIG_PATH}" ] || [ "${RETRIES}" -eq 0 ]; do
  sleep 1
  RETRIES=$((RETRIES - 1))
done

if [ -f "${KUBECONFIG_PATH}" ]; then
  echo "[✓] K3s cluster ready! Kubeconfig saved at: ${KUBECONFIG_PATH}"
  echo "[*] Run: kubectl --kubeconfig=${KUBECONFIG_PATH} get nodes"
else
  echo "[!] Failed to locate generated kubeconfig."
  exit 1
fi

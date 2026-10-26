# run_k3s_sandbox.ps1
# Spins up an ephemeral single-node K3s cluster using Podman on Windows
param (
    [string]$ClusterName = "k8s-sandbox-01",
    [int]$ApiPort = 6443,
    [string]$OutputDir = "$PSScriptRoot\..\k8s-output"
)

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

Write-Host "[*] Checking Podman machine status..."
$machineStatus = podman machine list --format "{{.Running}}"
if ($machineStatus -notcontains "true") {
    Write-Host "[!] Starting Podman machine..."
    podman machine start
}

Write-Host "[*] Removing existing container if present..."
podman rm -f $ClusterName 2>$null | Out-Null

Write-Host "[*] Launching ephemeral K3s cluster '$ClusterName' via Podman..."
# Using privileged mode, mounting volume for kubeconfig output
podman run -d --privileged `
    --name $ClusterName `
    -p "${ApiPort}:6443" `
    -e K3S_KUBECONFIG_OUTPUT=/output/kubeconfig.yaml `
    -e K3S_KUBECONFIG_MODE="666" `
    -v "${OutputDir}:/output:Z" `
    rancher/k3s:v1.31.0-k3s1 server --disable traefik --disable servicelb

Write-Host "[*] Waiting for kubeconfig to be generated in $OutputDir..."
$kubeconfigPath = Join-Path $OutputDir "kubeconfig.yaml"
$retries = 30
while ($retries -gt 0 -and -not (Test-Path $kubeconfigPath)) {
    Start-Sleep -Seconds 1
    $retries--
}

if (Test-Path $kubeconfigPath) {
    Write-Host "[✓] K3s cluster ready! Kubeconfig saved at: $kubeconfigPath"
    Write-Host "[*] Use: kubectl --kubeconfig=$kubeconfigPath get nodes"
} else {
    Write-Error "Failed to locate generated kubeconfig after 30 seconds."
}

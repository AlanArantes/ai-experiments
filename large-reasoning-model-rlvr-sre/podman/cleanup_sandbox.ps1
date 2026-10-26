# cleanup_sandbox.ps1
param (
    [string]$ClusterName = "k8s-sandbox-01",
    [string]$OutputDir = "$PSScriptRoot\..\k8s-output"
)

Write-Host "[*] Stopping and removing Podman container '$ClusterName'..."
podman rm -f $ClusterName 2>$null | Out-Null

$kubeconfigPath = Join-Path $OutputDir "kubeconfig.yaml"
if (Test-Path $kubeconfigPath) {
    Remove-Item $kubeconfigPath -Force
}
Write-Host "[✓] Sandbox cleaned up successfully."

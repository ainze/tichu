#requires -Version 5.1
<#
.SYNOPSIS
  Build the Tichu inference serving image (ADR-0027).

.DESCRIPTION
  Bakes a single exported model dir into a CPU-only serving image. All four
  difficulty tiers are served from that one model via skill-decile conditioning
  (easy stays the RuleAgent baseline). The weights live OUTSIDE the repo, so
  they are passed as a BuildKit named build-context rather than copied in.

.EXAMPLE
  ./docker/build.ps1
  # builds tichu-inference:bc-100k-v5 from C:\workbench\tichu\data\export\bc_full_100k_v5

.EXAMPLE
  ./docker/build.ps1 -Model awr_full_100k_v5_round -Tag tichu-inference:awr-100k-v5
  # NOTE: an AWR dir ships policy.pt only; the 3 standalone nets must be supplied too.
#>
[CmdletBinding()]
param(
    [string]$Model        = "bc_full_100k_v5",
    [string]$ExportRoot   = "C:\workbench\tichu\data\export",
    [string]$Tag          = "tichu-inference:bc-100k-v5",
    [string]$TorchVersion = "2.11.0"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$ModelPath = Join-Path $ExportRoot $Model

if (-not (Test-Path (Join-Path $ModelPath "policy.pt"))) {
    throw "No policy.pt under '$ModelPath'. Check -Model / -ExportRoot."
}

Write-Host "Building $Tag" -ForegroundColor Cyan
Write-Host "  model:  $Model ($ModelPath)"
Write-Host "  torch:  $TorchVersion (CPU)"

$env:DOCKER_BUILDKIT = "1"
docker build `
    -f (Join-Path $ScriptDir "Dockerfile") `
    --build-context "models=$ExportRoot" `
    --build-arg "MODEL=$Model" `
    --build-arg "TORCH_VERSION=$TorchVersion" `
    -t $Tag `
    $RepoRoot

if ($LASTEXITCODE -ne 0) { throw "docker build failed (exit $LASTEXITCODE)" }
Write-Host "Built $Tag" -ForegroundColor Green
Write-Host "Run with: docker run --rm -p 8000:8000 $Tag   (or: docker compose up -d)"

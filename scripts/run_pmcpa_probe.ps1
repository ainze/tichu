<#
  pMCPA A/B driver (ADR-0036) - RESUMABLE.

  Runs the shards sequentially, each saturating the box with -Workers internal
  processes. RESUMABILITY: each shard self-skips if its shard_NNNN.npz already
  exists, so re-launching this exact command after a crash/kill/reboot re-runs
  ONLY the missing shards. There is no separate resume flag - just relaunch.

  Defaults = the 48h probe (K=128, steps=3, n=4300 deals -> ~8600 seat-swap
  observations, CI ~ +/-4). For the 2-week decisive run, scale -NDeals 20000,
  -Worlds 256, -NShards 100 (see the bottom of this file).

  Launch DETACHED (survives the terminal / agent turn) - do NOT background it
  via the harness:

    Start-Process -FilePath "powershell.exe" `
      -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File", `
        "C:\workbench\tichu\.claude\worktrees\blissful-buck-91253d\scripts\run_pmcpa_probe.ps1" `
      -WorkingDirectory "C:\workbench\tichu\.claude\worktrees\blissful-buck-91253d" `
      -WindowStyle Hidden

  Monitor:  Get-Content <OutDir>\driver.log -Wait
  Pool:     python -m scripts.pool_pmcpa_ab --out-dir <OutDir>
#>
param(
  [string]$Worktree   = "C:\workbench\tichu\.claude\worktrees\blissful-buck-91253d",
  [string]$ThetaODir  = "C:\workbench\tichu\data\runs\cotrain_vine_v1\warm",
  [string]$Pool       = "C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet",
  [string]$Opponent   = "offline",   # "offline" = theta_o control; "export" = ship bar
  [string]$ExportDir  = "C:\workbench\tichu\data\runs\cotrain_wish_v5\export\iter_06225",
  [string]$OutDir     = "C:\workbench\tichu\data\runs\pmcpa_probe\ab_offline",
  [int]$NDeals        = 4300,
  [int]$NShards       = 20,
  [int]$Workers       = 10,
  [int]$Worlds        = 128,
  [int]$DecisionsPerWorld = 4,
  [int]$Branches      = 4,
  [int]$Steps         = 3,
  [double]$Lr         = 0.01,
  [double]$KlCoef     = 1.0,
  [switch]$AdaptTrunk
)

# Prepend THIS worktree's src so 'import tichu_*' does not resolve to a sibling
# worktree's editable install (the known shadowing gotcha).
$env:PYTHONPATH = "$Worktree\src;$env:PYTHONPATH"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "driver.log"
"[{0}] pMCPA A/B start: opponent={1} NDeals={2} NShards={3} Workers={4} K={5} steps={6} adaptTrunk={7}" -f `
  (Get-Date -Format s), $Opponent, $NDeals, $NShards, $Workers, $Worlds, $Steps, [bool]$AdaptTrunk |
  Tee-Object -FilePath $log -Append

for ($k = 0; $k -lt $NShards; $k++) {
  "[{0}] shard {1}/{2}" -f (Get-Date -Format s), $k, $NShards |
    Tee-Object -FilePath $log -Append
  $shardArgs = @(
    "-m","scripts.run_pmcpa_ab",
    "--theta-o-dir",$ThetaODir,"--opponent",$Opponent,"--export-dir",$ExportDir,
    "--pool",$Pool,"--out-dir",$OutDir,
    "--shard",$k,"--n-shards",$NShards,"--n-deals",$NDeals,"--workers",$Workers,
    "--worlds",$Worlds,"--decisions-per-world",$DecisionsPerWorld,"--branches",$Branches,
    "--steps",$Steps,"--lr",$Lr,"--kl-coef",$KlCoef
  )
  if ($AdaptTrunk) { $shardArgs += "--adapt-trunk" }
  & python @shardArgs *>> $log
  if ($LASTEXITCODE -ne 0) {
    "[{0}] shard {1} FAILED (exit {2}); relaunch to resume." -f `
      (Get-Date -Format s), $k, $LASTEXITCODE | Tee-Object -FilePath $log -Append
    exit $LASTEXITCODE
  }
}

"[{0}] all {1} shards done. Pool: python -m scripts.pool_pmcpa_ab --out-dir {2}" -f `
  (Get-Date -Format s), $NShards, $OutDir | Tee-Object -FilePath $log -Append

# --- 2-week decisive run (after a positive probe) -----------------------------
#   ...run_pmcpa_probe.ps1 -NDeals 20000 -NShards 100 -Worlds 256 -OutDir ...\ab_offline_decisive
# Ship bar (vs the shipped iter_06225) - same, with:
#   -Opponent export -OutDir ...\ab_export

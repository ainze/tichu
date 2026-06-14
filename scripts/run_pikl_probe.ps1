<#
  piKL A/B λ-sweep driver (ADR-0037) - RESUMABLE.

  For each λ in -Lambdas, runs the full sharded seat-swap A/B of piKL(τ) vs the
  un-searched iter_06225 export (the mechanism read AND the ship bar - they
  collapse, since τ IS the export). Each (λ, shard) self-skips if its
  shard_NNNN.npz exists, so re-launching this exact command resumes only the
  missing work. There is no resume flag - just relaunch.

  Defaults = the PROBE (worlds=10, n=4000 deals -> ~8000 seat-swap obs, CI ~ +/-4)
  across the full λ grid. For the DECISIVE run after a positive probe, scale
  -Worlds 20 -NDeals 20000 -NShards 100 -OutRoot ...\pikl_decisive.

  Run in the FOREGROUND (live progress on the console; Ctrl-C to stop, relaunch to
  resume - each (λ, shard) self-skips if its shard_NNNN.npz exists):

    cd C:\workbench\tichu\.claude\worktrees\competent-stonebraker-33120c
    .\scripts\run_pikl_probe.ps1

  Output also tees to <OutRoot>\driver.log.
  Pool one λ:  python -m scripts.pool_pmcpa_ab --out-dir <OutRoot>\lam_0p1
#>
param(
  [string]$Worktree  = "C:\workbench\tichu\.claude\worktrees\competent-stonebraker-33120c",
  [string]$ExportDir = "C:\workbench\tichu\data\runs\cotrain_wish_v5\export\iter_06225",
  [string]$Pool      = "C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet",
  [string]$OutRoot   = "C:\workbench\tichu\data\runs\pikl_probe",
  [double[]]$Lambdas = @(0.03, 0.1, 0.3, 1.0),
  [int]$NDeals       = 4000,
  [int]$NShards      = 20,
  [int]$Workers      = 11,
  [int]$Worlds       = 10,
  [int]$K            = 5,    # gate: max alt-rank 4, so k=5 covers all 663 corrections
  [double]$QScale    = 22.0,
  [int]$SkillDecile  = 9
)

# Prepend THIS worktree's src so 'import tichu_*' does not resolve to a sibling
# worktree's editable install (the known shadowing gotcha).
$env:PYTHONPATH = "$Worktree\src;$env:PYTHONPATH"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$log = Join-Path $OutRoot "driver.log"
"[{0}] piKL A/B sweep start: lambdas={1} NDeals={2} NShards={3} Workers={4} worlds={5} k={6} qscale={7}" -f `
  (Get-Date -Format s), ($Lambdas -join ","), $NDeals, $NShards, $Workers, $Worlds, $K, $QScale |
  Tee-Object -FilePath $log -Append

foreach ($lam in $Lambdas) {
  $tag = ("{0}" -f $lam).Replace(".", "p")
  $outDir = Join-Path $OutRoot "lam_$tag"
  "[{0}] === lambda={1} -> {2} ===" -f (Get-Date -Format s), $lam, $outDir |
    Tee-Object -FilePath $log -Append
  $shardArgs = @(
    "-m","scripts.run_pikl_ab",
    "--export-dir",$ExportDir,"--pool",$Pool,"--out-dir",$outDir,
    "--n-shards",$NShards,"--n-deals",$NDeals,"--workers",$Workers,
    "--worlds",$Worlds,"--k",$K,"--lam",$lam,"--q-scale",$QScale,
    "--skill-decile",$SkillDecile
  )
  & python @shardArgs *>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) {
    "[{0}] lambda={1} FAILED (exit {2}); relaunch to resume." -f `
      (Get-Date -Format s), $lam, $LASTEXITCODE | Tee-Object -FilePath $log -Append
    exit $LASTEXITCODE
  }
}

"[{0}] sweep done. Pool each: python -m scripts.pool_pmcpa_ab --out-dir {1}\lam_<tag>" -f `
  (Get-Date -Format s), $OutRoot | Tee-Object -FilePath $log -Append

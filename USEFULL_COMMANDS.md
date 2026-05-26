pip install -e . --force-reinstall --no-deps


python -m tichu_training.cli.parse_bsw `
    --archive C:/workbench/tichu/data/archive.zst `
    --output C:/workbench/tichu/data/parquet_50k `
    --subset 50000 `
    --workers 8


python -m tichu_training.cli.parse_bsw `
    --archive C:\workbench\tichu\data\archive.zst `
    --output C:\workbench\tichu\data\parquet_1 `
    --subset 100 --workers 12 -v

# Quick group-by-mode count
Get-Content C:/workbench/tichu/data/parquet_smoke/failure_details.tsv `
  | Select-Object -Skip 1 `
  | ForEach-Object { ($_ -split "`t")[2] } `
  | Group-Object | Sort-Object Count -Descending


python -m tichu_training.cli.compute_trueskill `
     --input C:\workbench\tichu\data\archive.zst `
     --output C:\workbench\tichu\data\ratings_100k.parquet `
     --subset 100000 `
     --min-games 20 -v


python -m tichu_training.cli.train_bc `
    --config configs/bc_smoke_parquet.yaml `
    --run-dir C:\workbench\tichu\data\runs\bc_smoke_100k_par `
    --workers 11 -v
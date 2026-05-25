

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



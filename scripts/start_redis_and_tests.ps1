param(
    [string]$RedisUrl = "redis://127.0.0.1:6379/0"
)

Write-Host "Starting Redis via docker compose..."
docker compose up -d

Write-Host "Installing backend in editable mode..."
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $python -m pip install -e ..\backend

Write-Host "Running Redis-backed test..."
$env:REDIS_URL = $RedisUrl
& $python -m pytest ..\backend\tests\test_redis_job_store.py -q -s

@echo off
REM Helper: set REDIS_URL and run a command in this shell
if "%1"=="" (
  echo Usage: %0 "command to run"
  echo Example: %0 "python -m uvicorn src.api.main:app --reload"
  exit /b 1
)

set REDIS_URL=redis://localhost:6379/0
echo Set REDIS_URL=%REDIS_URL%

%*

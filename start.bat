@echo off
:: ═══════════════════════════════════════════════════════════
::  GUARDRAILS LOCAL RAG BOT — Start Script (Windows)
:: ═══════════════════════════════════════════════════════════
echo.
echo  ╔═══════════════════════════════════════╗
echo  ║   GUARDRAILS LOCAL RAG BOT v1.0       ║
echo  ║   FastAPI + Vanilla JS Web App        ║
echo  ╚═══════════════════════════════════════╝
echo.

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] .venv not found. Using system Python.
)

:: Install dependencies if not present
python -c "import uvicorn" 2>nul || (
    echo [INFO] Installing backend dependencies...
    pip install -r backend\requirements.txt
)

echo [INFO] Starting FastAPI server on http://localhost:8000
echo [INFO] Press Ctrl+C to stop.
echo.

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

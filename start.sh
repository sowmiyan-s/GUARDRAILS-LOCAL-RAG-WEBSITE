#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  GUARDRAILS LOCAL RAG BOT — Start Script (macOS / Linux)
# ═══════════════════════════════════════════════════════════
set -e

echo ""
echo " ╔═══════════════════════════════════════╗"
echo " ║   GUARDRAILS LOCAL RAG BOT v1.0       ║"
echo " ║   FastAPI + Vanilla JS Web App        ║"
echo " ╚═══════════════════════════════════════╝"
echo ""

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[WARN] Virtual environment not found. Using system Python."
fi

# Install dependencies if missing
python -c "import uvicorn" 2>/dev/null || {
    echo "[INFO] Installing backend dependencies..."
    pip install -r backend/requirements.txt
}

PORT=${PORT:-8000}
echo "[INFO] Starting FastAPI server on http://localhost:$PORT"
echo "[INFO] Press Ctrl+C to stop."
echo ""


exec uvicorn backend.main:app --host 0.0.0.0 --port $PORT
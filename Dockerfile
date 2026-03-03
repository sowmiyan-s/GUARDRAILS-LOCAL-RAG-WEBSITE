# Guardrails Local RAG Bot — Docker Image
#
# Build:   docker build -t rag-bot .
# Run:     docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway rag-bot
#
# The container serves the FastAPI backend + pre-bundled frontend on port 8000.
# Ollama must be running on the HOST machine (or in a linked container) and the
# OLLAMA_HOST env-var should point to it:
#   docker run -e OLLAMA_HOST=http://host.docker.internal:11434 -p 8000:8000 rag-bot

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached unless requirements change)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy frontend (served as static files by FastAPI)
COPY frontend/ ./frontend/

# Copy guardrails config
COPY guardrails_config/ ./guardrails_config/

# FAISS vector store will be written here at runtime
RUN mkdir -p .faiss_storage

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

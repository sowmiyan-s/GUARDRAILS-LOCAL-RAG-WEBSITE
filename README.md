<div align="center">

<br/>

<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
&nbsp;
<img src="https://img.shields.io/badge/LangChain-RAG-311b92?style=for-the-badge&logo=chainlink&logoColor=white"/>
&nbsp;
<img src="https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama&logoColor=white"/>
&nbsp;
<img src="https://img.shields.io/badge/FAISS-Vector%20Store-0064A4?style=for-the-badge&logo=meta&logoColor=white"/>

<br/><br/>

# GUARDRAILS LOCAL RAG BOT

### A privacy-first, fully offline AI document assistant — secured by a tiered safety guardrails system

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3b82f6?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Offline](https://img.shields.io/badge/Mode-100%25%20Offline-76b900?style=flat-square)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-a78bfa?style=flat-square)](#contributing)

<br/>

> Upload any document. Ask anything. Get answers — **entirely on your machine.**  
> No cloud. No API keys. No data leaves your device.

<br/>

</div>

---

## Architecture

```
User browser → FastAPI (port 8000) → Tiered Safety Input Check
                                    → LangChain History Retriever
                                    → FAISS Vector Store (local disk)
                                    → Ollama Local LLM
                                    → Tiered Safety Output Check
                                    → Response
```

---

## Why This Project?

Most RAG chatbots rely on cloud APIs, which creates **privacy risks** for sensitive documents — contracts, medical records, internal reports. This project solves that by:

- Running the **LLM locally** via Ollama (no data transmitted)
- Embedding documents **offline** using HuggingFace sentence-transformers
- Enforcing **tiered safety policies** with 4 sensitivity levels
- Serving everything through a **single FastAPI server** with a VanillaJS frontend

---

## Feature Highlights

<table>
<tr>
<td width="50%">

### Core
- **100% Offline** — zero external network calls at runtime
- **Multi-format ingestion** — PDF, TXT, DOCX
- **Persistent FAISS cache** — same file re-uploads skip re-embedding
- **Multi-turn conversation** — full history-aware retrieval
- **Any Ollama model** — Gemma, Llama3, Mistral, Phi, and more

</td>
<td width="50%">

### Safety
- **4-Tier Data Sensitivity System** — Public → Internal → Confidential → Restricted
- **Jailbreak / prompt injection detection** — always active
- **Credential & API key protection** — Internal+
- **PII protection** — SSN, email, phone, DOB, credit card (Confidential+)
- **Regulated data guards** — HIPAA / GDPR / financial categories (Restricted)

</td>
</tr>
</table>

---

## Data Sensitivity Levels

| Level | Badge | What is Protected |
|---|---|---|
| **Public** | ![](https://img.shields.io/badge/-Public-22c55e?style=flat-square) | Jailbreak & prompt injection only |
| **Internal** | ![](https://img.shields.io/badge/-Internal-3b82f6?style=flat-square) | + API keys, credentials, passwords, tokens |
| **Confidential** | ![](https://img.shields.io/badge/-Confidential-eab308?style=flat-square) | + SSN, email, phone number, DOB, credit card |
| **Restricted** | ![](https://img.shields.io/badge/-Restricted-ef4444?style=flat-square) | + Medical records, diagnoses, financials, HIPAA/GDPR |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web UI** | Vanilla JS + HTML/CSS (served by FastAPI) |
| **API Server** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| **LLM Engine** | [Ollama](https://ollama.com) — local model inference |
| **Embeddings** | [HuggingFace](https://huggingface.co) `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) — disk-persisted |
| **RAG Pipeline** | [LangChain](https://langchain.com) — retrieval chains + chat history |
| **Safety Rails** | Custom tiered guardrails system (input + output) |

---

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running locally
- At least one model pulled via Ollama:

```bash
ollama pull gemma3:1b
# or any other model: llama3.1, phi3, mistral, etc.
```

---

## Quick Start

### Option 1 — Local (Windows)

```bat
start.bat
```

### Option 2 — Local (macOS / Linux)

```bash
chmod +x start.sh
./start.sh
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Option 3 — Manual setup

```bash
# 1. Clone the repository
git clone https://github.com/sowmiyan-s/guardrails-local-rag-bot.git
cd guardrails-local-rag-bot

# 2. Create & activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt

# 4. Start the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 4 — Docker

```bash
# Build and run (Ollama must be running on the host)
docker compose up --build
```

Or without Compose:

```bash
docker build -t rag-bot .
docker run -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  rag-bot
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Ollama status, installed models, sensitivity profiles |
| `POST` | `/api/ollama/start` | Attempt to start the local Ollama process |
| `POST` | `/api/upload` | Upload documents and build the RAG chain |
| `POST` | `/api/chat` | Send a question, receive a guardrail-checked answer |
| `POST` | `/api/clear` | Clear conversation history for a session |
| `GET` | `/` | Serve the frontend web app |

Interactive API docs available at **[http://localhost:8000/docs](http://localhost:8000/docs)** (Swagger UI).

---

## Project Structure

```
guardrails-local-rag-bot/
│
├── backend/
│   ├── main.py               # FastAPI application — all API routes & RAG logic
│   ├── __init__.py
│   └── requirements.txt      # Backend Python dependencies
│
├── frontend/
│   ├── index.html            # Single-page web app
│   └── static/
│       ├── style.css         # UI styles
│       └── app.js            # Frontend logic
│
├── guardrails_config/        # Safety rail configuration (colang + YAML)
│   ├── config.yml
│   ├── prompts.yml
│   ├── safety.co
│   └── off_topic.co
│
├── assets/
│   └── architecture.svg      # System architecture diagram
│
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # One-command Docker setup
├── Procfile                  # Render / Railway deployment
├── runtime.txt               # Python version pin for PaaS
├── requirements.txt          # Root-level requirements (mirrors backend/)
├── start.bat                 # Windows quick-start script
├── start.sh                  # macOS/Linux quick-start script
├── .env.example              # Environment variable template
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

> `.faiss_storage/` is auto-generated on first document upload and excluded from version control (and Docker builds).

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `NO_PROXY` | `huggingface.co,...` | Bypass proxy for local+HF calls |
| `PORT` | `8000` | Server port (auto-set by PaaS) |

### Chunking Parameters

Adjustable per-session via the sidebar in the UI:
- **Chunk Size** (default 1000 chars)
- **Chunk Overlap** (default 200 chars)

Different chunk settings for the same file produce a separate FAISS index automatically.

---

## Deployment

### Render

1. Connect your GitHub repository to [Render](https://render.com)
2. Choose **Web Service** → **Python**
3. Set **Build Command**: `pip install -r backend/requirements.txt`
4. Set **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add env var: `OLLAMA_HOST` → your Ollama endpoint

> **Note:** This app uses a **local Ollama instance** for LLM inference. For cloud deployment you need either a self-hosted Ollama server or a compatible OpenAI-style API endpoint.

### Railway

```bash
railway login
railway init
railway up
```

The `Procfile` is picked up automatically.

### Docker (self-hosted)

```bash
docker compose up -d
```

---

## Contributing

Contributions are welcome. Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feat/your-feature`
3. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/): `git commit -m "feat: add X"`
4. **Push** and open a **Pull Request**

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/sowmiyan-s/guardrails-local-rag-bot/issues).

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ by **[Sowmiyan S](https://github.com/sowmiyan-s)**

*FastAPI · LangChain · Ollama · HuggingFace · FAISS · Vanilla JS*

</div>

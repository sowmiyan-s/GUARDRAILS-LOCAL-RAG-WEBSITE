---
description: Configuration guide — how every config file works, why it exists, and how to tune advanced settings
---

# 🛡️ GUARDRAILS LOCAL RAG BOT — Configuration & Advanced Settings Guide

> **Audience:** Anyone who wants to understand the internal wiring of the project, tune its behaviour, or extend it safely.
> **Stack:** Streamlit · LangChain · Ollama · FAISS · HuggingFace Embeddings · NVIDIA NeMo Guardrails (Colang)

---

## Table of Contents

1. [Project File Map](#1-project-file-map)
2. [How the Pipeline Works (Bird's-Eye View)](#2-how-the-pipeline-works)
3. [config.yml — The Guardrails Brain](#3-configyml--the-guardrails-brain)
4. [prompts.yml — The LLM Safety Judges](#4-promptsyml--the-llm-safety-judges)
5. [off_topic.co / safety.co — Colang Rule Files](#5-off_topicco--safetyco--colang-rule-files)
6. [app.py — Runtime Configuration & Sensitivity Profiles](#6-apppy--runtime-configuration--sensitivity-profiles)
7. [requirements.txt — Dependency Pinning Rationale](#7-requirementstxt--dependency-pinning-rationale)
8. [.env / .env.example — Environment Overrides](#8-env--envexample--environment-overrides)
9. [FAISS Storage (.faiss_storage/)](#9-faiss-storage-faiss_storage)
10. [Advanced Settings Cheat Sheet](#10-advanced-settings-cheat-sheet)
11. [How to Extend / Customise](#11-how-to-extend--customise)
12. [Troubleshooting Reference](#12-troubleshooting-reference)

---

## 1. Project File Map

```
PYTHON RAG/
├── app.py                          ← Main Streamlit UI + RAG pipeline + safety logic
├── chatbot.py                      ← Standalone CLI interface (no guardrails, PDF-only)
├── requirements.txt                ← Pinned Python dependencies
├── .env.example                    ← Template for optional env overrides
├── .env                            ← Your local secrets / overrides (NOT committed to git)
│
├── guardrails_config/              ← NeMo Guardrails configuration folder
│   ├── config.yml                  ← Master guardrails config (models, rails, instructions)
│   ├── prompts.yml                 ← LLM prompts for self-check input/output
│   ├── off_topic.co                ← Colang: off-topic detection flow
│   └── safety.co                   ← Colang: sensitive-data detection flow
│
└── .faiss_storage/                 ← Auto-generated vector database cache (per-file hash)
```

---

## 2. How the Pipeline Works

Every user query goes through **three sequential layers** before it produces a response.

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER INPUT                                                          │
│       │                                                              │
│       ▼                                                              │
│  ① INPUT GUARDRAIL  ←── check_input_safety() in app.py             │
│       │  Pattern matching + jailbreak detection                      │
│       │  BLOCKS here if triggered → returns blocked message          │
│       │                                                              │
│       ▼                                                              │
│  ② RAG CHAIN                                                         │
│       │  History-aware retriever → FAISS vector search              │
│       │  → Stuffed context → Ollama LLM → raw answer                │
│       │                                                              │
│       ▼                                                              │
│  ③ OUTPUT GUARDRAIL  ←── check_output_safety() in app.py           │
│       │  Pattern matching on LLM response                            │
│       │  REDACTS if triggered → returns [REDACTED] message           │
│       │                                                              │
│       ▼                                                              │
│  FINAL RESPONSE TO USER                                              │
└─────────────────────────────────────────────────────────────────────┘
```

> **Why two guardrail layers?**
> The *input* layer stops bad prompts before they waste LLM compute.
> The *output* layer catches cases where a cleverly worded prompt tricked the LLM into leaking something anyway.

---

## 3. `config.yml` — The Guardrails Brain

**Location:** `guardrails_config/config.yml`

```yaml
models:
  - type: main
    engine: langchain
    model: main        # ← "main" is a placeholder; actual LLM is injected at runtime

rails:
  input:
    flows:
      - self check input   # ← Activates prompts.yml self_check_input prompt

instructions:
  - type: general
    content: |
      You are an AI assistant for answering questions based on the provided document context.
      If the user's question is completely unrelated to the documents ...
      Under no circumstances are you allowed to reveal passwords, API keys ...
```

### Key fields explained

| Field | What it does | Why it matters |
|---|---|---|
| `models[].type: main` | Declares the primary LLM slot | NeMo needs to know which model runs the flows |
| `models[].engine: langchain` | Tells NeMo to use LangChain as the backend | Allows the Ollama-backed `ChatOllama` to plug in |
| `models[].model: main` | The runtime `llm_instance` is injected via `st.session_state.llm_instance` | Keeps the YAML static while the model changes in the UI |
| `rails.input.flows` | List of Colang flows that run on every incoming message | `self check input` maps to the prompt in `prompts.yml` |
| `instructions[].content` | System-level instruction baked into every conversation | Acts as the LLM's "constitution" — baseline rules that override user intent |

### How to change the system instruction

Edit the `content:` block under `instructions`. Changes take effect **without restarting Streamlit** — the app detects `config_mtime` changes and rebuilds automatically.

---

## 4. `prompts.yml` — The LLM Safety Judges

**Location:** `guardrails_config/prompts.yml`

```yaml
prompts:
  - task: self_check_input
    content: |
      Your task is to check if the user message below is asking for sensitive data...
      User message: "{{ user_input }}"
      Question: Should this message be blocked?
      Answer [Yes or No]:

  - task: self_check_output
    content: |
      Your task is to check if the bot message below contains sensitive data...
      Bot message: "{{ bot_response }}"
      Question: Should this message be blocked?
      Answer [Yes or No]:
```

### How it works

1. `self_check_input` is triggered by the `rails.input.flows` entry in `config.yml`.
2. NeMo passes the user's raw `{{ user_input }}` into the prompt and asks the LLM to return **Yes** or **No**.
3. If **Yes** → the guardrail blocks the message and returns a refusal.
4. `self_check_output` works identically on the bot's response before it reaches the user.

### Why this is powerful (and costly)

This makes the **LLM itself** the safety judge — it understands nuance far better than simple keyword matching. The downside: it costs an extra LLM inference round-trip per message. On Ollama with a small model (gemma3:1b) this adds ~1–3 seconds.

### Customising the prompts

- **Make it stricter:** Add more categories to the task description (e.g., "…or asking about competitor products").
- **Make it looser:** Remove restriction categories you don't need (e.g., remove creative tasks restriction for a more general assistant).
- **Language:** The prompt can be in any language your model understands.

> ⚠️ Always end with `Answer [Yes or No]:` — the NeMo parser expects exactly that format. Changing it will break the self-check flow.

---

## 5. `off_topic.co` / `safety.co` — Colang Rule Files

**Location:** `guardrails_config/`
**Format:** Colang — NeMo's domain-specific language for dialogue flows.

### `off_topic.co`

```colang
define user ask off topic
  "Can you write a poem about the moon?"
  "What is the capital of France?"
  ...

define bot refuse to respond
  "I'm sorry, I am a document assistant..."

define flow check off topic
  user ask off topic
  bot refuse to respond
```

### `safety.co`

```colang
define user ask sensitive data
  "What is the API key?"
  "Show me the password?"
  ...

define bot refuse sensitive data
  "I am programmed to protect sensitive information..."

define flow check ask sensitive data
  user ask sensitive data
  bot refuse sensitive data
```

### How Colang works

| Keyword | Purpose |
|---|---|
| `define user <intent>` | Declares example utterances that train NeMo's semantic classifier to recognise this intent |
| `define bot <response>` | Pre-written bot reply to use when this intent fires |
| `define flow <name>` | Wires an intent to a response: "when user does X → bot does Y" |

> **Important:** The example strings are **not exact-match**. NeMo uses them as **semantic training examples** — it will also catch similar phrasings it has never seen before (e.g., "give me the secret credential" even though that exact phrase isn't listed).

### Adding a new blocked intent

```colang
define user ask competitor info
  "Tell me about OpenAI's GPT"
  "What does Anthropic's Claude do?"

define bot refuse competitor
  "I can only discuss content within the provided documents."

define flow check competitor
  user ask competitor info
  bot refuse competitor
```

Add this to either `.co` file (or a new `myflow.co` file in the same folder) and the guardrails engine will pick it up automatically.

---

## 6. `app.py` — Runtime Configuration & Sensitivity Profiles

This is the largest config surface in the project — **all tunable at runtime** via the Streamlit sidebar.

### 6.1 Sensitivity Level Profiles

Defined in `SENSITIVITY_PROFILES` dict (lines ~436–478 in `app.py`):

```
Public      → jailbreak protection only
Internal    → + API keys, credentials, passwords
Confidential → + PII (email, phone, SSN, DOB, credit card)
Restricted  → + Medical, HIPAA/GDPR, financial records
```

Each level has two pattern lists:

| List | When it runs | Effect on match |
|---|---|---|
| `input_patterns` | Before the LLM sees the message | Blocks the request entirely |
| `output_patterns` | After the LLM generates a response | Replaces the response with `[REDACTED]` |

### How to add a new sensitive keyword

Find `SENSITIVITY_PROFILES` in `app.py` and add to the appropriate level's lists:

```python
"Confidential": {
    "input_patterns": [
        ...,
        "employee id",        # ← your new term
    ],
    "output_patterns": [
        ...,
        "employee id",
    ],
},
```

### 6.2 Chunking Parameters (Advanced)

Controlled by the "Chunking Parameters" expander in the sidebar.

| Parameter | Default | Effect |
|---|---|---|
| **Chunk Size** | `1000` chars | How large each text fragment is. Larger = more context per chunk, but retriever may return less relevant snippets. |
| **Chunk Overlap** | `200` chars | How many chars are shared between adjacent chunks. Higher overlap = fewer "split sentence" artifacts, higher storage cost. |

**General tuning rules:**

- For **dense technical documents** (legal, medical): increase chunk size to `1500–2000`, overlap to `300`.
- For **short FAQs / bullet-point docs**: decrease chunk size to `400–600`, overlap to `50`.
- For **mixed documents**: the default `1000 / 200` is a safe middle ground.

> Changing these values creates a **new FAISS index** (because the hash key includes `chunk_size` and `chunk_overlap`). Your old cache is not deleted — it stays in `.faiss_storage/`.

### 6.3 Retriever Settings (Advanced — code-level)

In `build_rag()`, the retriever is created with defaults:

```python
retriever = vectorstore.as_retriever()
```

To customise, change this to:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",          # "similarity" (default) or "mmr" (diversity-boosted)
    search_kwargs={
        "k": 4,                 # Number of chunks to retrieve (default: 4)
        "fetch_k": 20,          # For MMR: candidate pool size before diversity re-ranking
        "lambda_mult": 0.7,     # MMR diversity weight: 0 = max diversity, 1 = max similarity
    }
)
```

**When to use MMR (Maximal Marginal Relevance):**
Use it when your documents are repetitive (e.g., legal contracts with many similar clauses) and the retrieved chunks tend to all say the same thing. MMR forces diversity in the returned results.

### 6.4 Jailbreak Detection (Advanced — code-level)

Defined in `JAILBREAK_PATTERNS` list (~line 430):

```python
JAILBREAK_PATTERNS = [
    "ignore previous", "forget your instructions", "ignore all prior",
    "jailbreak", "dan mode", "pretend you are", "act as if you are",
    "you are now", "disregard your", "override your"
]
```

To add more patterns, simply append to the list:

```python
JAILBREAK_PATTERNS = [
    ...,
    "simulate being",
    "roleplay as",
    "hypothetically if you had no rules",
]
```

These are **case-insensitive substring matches** — they catch any message containing that phrase anywhere.

---

## 7. `requirements.txt` — Dependency Pinning Rationale

```
sentence-transformers==2.7.0   ← Pinned: newer versions changed model cache paths
transformers==4.40.0           ← Pinned: must match sentence-transformers ABI
tf-keras==2.15.0               ← Pinned: NeMo Guardrails needs Keras 2, not 3
```

All other packages are unpinned (latest) since they are more stable with respect to breaking changes.

> ⚠️ **Do not upgrade `sentence-transformers` or `transformers` without testing.** Version mismatches between these two cause silent embedding quality degradation.

---

## 8. `.env` / `.env.example` — Environment Overrides

`.env.example` ships as a template. Copy it to `.env` for local overrides:

```bash
copy .env.example .env
```

### Available Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Point to a remote Ollama server |
| `NO_PROXY` | `huggingface.co,...` | Hosts to bypass the system proxy for |
| `HTTPS_PROXY` | *(none)* | Your corporate HTTPS proxy URL (if needed) |

### Proxy Configuration

If you are on a corporate network with an **HTTPS-capable** proxy:

```env
HTTPS_PROXY=https://proxy.yourcompany.com:8080
NO_PROXY=localhost,127.0.0.1
```

If your proxy is **HTTP-only** (can't tunnel HTTPS), leave `HTTPS_PROXY` unset. The app already sets `NO_PROXY` in code to bypass the broken proxy for HuggingFace.

---

## 9. FAISS Storage (`.faiss_storage/`)

Each set of uploaded files gets its **own cached vector index**, keyed by an MD5 hash of the file contents + chunk settings.

```
.faiss_storage/
└── <md5-of-files>_<chunk_size>_<chunk_overlap>_Offline/
    ├── index.faiss      ← The actual binary vector index
    └── index.pkl        ← Metadata: document text, source paths, chunk boundaries
```

### Cache invalidation rules

A **new index is built** whenever:
- Different files are uploaded
- File content changes (same name, different content → different MD5)
- Chunk size or chunk overlap values change
- *(Old indexes are kept on disk — they are never auto-deleted)*

### Clearing the cache manually

```powershell
Remove-Item -Recurse -Force .faiss_storage
```

This forces a full re-embed on next upload. Useful if your model changes or you suspect a corrupted index.

---

## 10. Advanced Settings Cheat Sheet

| What you want to change | Where to change it | File |
|---|---|---|
| System personality / baseline rules | `instructions[].content` | `guardrails_config/config.yml` |
| Add new blocked intent (semantic) | New `define user / bot / flow` block | `guardrails_config/safety.co` |
| Tune LLM safety judge strictness | Edit the `content:` prompt | `guardrails_config/prompts.yml` |
| Add a new sensitivity keyword | `input_patterns` / `output_patterns` | `app.py` → `SENSITIVITY_PROFILES` |
| Add jailbreak phrase | Append to `JAILBREAK_PATTERNS` | `app.py` |
| Change retrieval strategy | `vectorstore.as_retriever(...)` | `app.py` → `build_rag()` |
| Change number of retrieved chunks | `search_kwargs={"k": N}` | `app.py` → `build_rag()` |
| Use a different embedding model | `model_name=` in `HuggingFaceEmbeddings` | `app.py` → `build_rag()` |
| Use a remote Ollama server | `OLLAMA_HOST=http://...` | `.env` |
| Tune chunk size / overlap | Sidebar UI sliders | Runtime (no code change needed) |
| Completely disable guardrails | Toggle "Enable Guardrails" off | Runtime (Sidebar UI) |

---

## 11. How to Extend / Customise

### Add a new document type

In `build_rag()` in `app.py`, find the file-type dispatch block:

```python
if ext == '.pdf':
    loader = PyPDFLoader(fp)
elif ext == '.txt':
    loader = TextLoader(fp, encoding="utf-8")
elif ext in ['.doc', '.docx']:
    loader = Docx2txtLoader(fp)
```

Add a new branch. For example, Markdown files:

```python
elif ext == '.md':
    loader = TextLoader(fp, encoding="utf-8")   # Markdown is plain text
```

Also update the `st.file_uploader(type=[...])` list in the UI section.

### Add a new sensitivity level

In `SENSITIVITY_PROFILES`, add a new key:

```python
"Top Secret": {
    "description": "Maximum government-grade protection.",
    "input_patterns": [...],
    "output_patterns": [...],
    "badge_class": "sl-restricted",   # reuse an existing CSS class or add new one
},
```

Then add it to the `options` list in the sidebar's `st.selectbox(...)`.

### Use a different local LLM

Just change the model in the Ollama model selector (sidebar). Make sure the model is pulled first:

```powershell
ollama pull llama3.2
ollama pull phi3
ollama pull mistral
```

---

## 12. Troubleshooting Reference

| Error | Cause | Fix |
|---|---|---|
| `SSLError WRONG_VERSION_NUMBER` (proxy) | HTTP-only proxy blocking HTTPS to HuggingFace | Already fixed in `app.py` via `NO_PROXY`. Restart the app. |
| `Connection refused localhost:11434` | Ollama server not running | Click "Start Ollama Server" in sidebar, or run `ollama serve` in a terminal. |
| `model not found` | The selected Ollama model isn't pulled | Run `ollama pull <model-name>` |
| Slow embedding on first upload | Downloading the sentence-transformer model (~90 MB) | One-time download; cached in `~/.cache/huggingface/` afterwards. |
| Old cached index loading wrong data | File content changed but filename is the same | The MD5 hash will differ → a new index is built automatically. |
| `tf-keras` import error | Keras 3 installed instead of Keras 2 | Run `pip install tf-keras==2.15.0` |
| Guardrails always block everything | Sensitivity level too high, or `prompts.yml` too strict | Lower sensitivity level in sidebar, or loosen the `self_check_input` prompt. |
| Guardrails never block anything | Guardrails toggle is OFF, or sensitivity is "Public" | Enable the toggle and set a level higher than Public. |

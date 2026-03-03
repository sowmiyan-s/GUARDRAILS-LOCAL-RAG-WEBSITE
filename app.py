import os
import tempfile
import hashlib
import nest_asyncio
import urllib.request
import json
import subprocess
import time

# Fix OMP error for FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Proxy fix ────────────────────────────────────────────────────────────────
# If a system HTTP proxy is configured, it cannot tunnel HTTPS traffic.
# Bypass the proxy entirely for HuggingFace and localhost so that model
# downloads / Ollama calls work correctly even behind a corporate HTTP proxy.
_NO_PROXY_HOSTS = "huggingface.co,*.huggingface.co,localhost,127.0.0.1"
os.environ.setdefault("NO_PROXY",   _NO_PROXY_HOSTS)
os.environ.setdefault("no_proxy",   _NO_PROXY_HOSTS)
# If you are behind a proxy that does support HTTPS, set HTTPS_PROXY in your
# .env file (e.g. HTTPS_PROXY=https://proxy.example.com:8080) and remove
# or adjust the lines above.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
from dotenv import load_dotenv

nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- Ollama Utilities ---
def is_ollama_running():
    try:
        urllib.request.urlopen("http://localhost:11434/", timeout=2)
        return True
    except Exception:
        return False

def get_installed_models():
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        data = json.loads(req.read().decode('utf-8'))
        return [model['name'] for model in data.get('models', [])]
    except Exception:
        return []

def start_ollama():
    try:
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(['ollama', 'serve'], creationflags=creationflags)
        for _ in range(20):
            if is_ollama_running():
                return True
            time.sleep(0.5)
        return False
    except Exception:
        return False

# Streamlit UI Configuration
st.set_page_config(
    page_title="GUARDRAILS LOCAL RAG BOT",
    page_icon="https://www.nvidia.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Background ---- */
.stApp {
    background-color: #080c14;
    color: #c9d1d9;
}

/* ---- Header bar ---- */
.top-header {
    background: #0d1117;
    border-bottom: 1px solid #1e2a3a;
    padding: 1.1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
}
.top-header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.logo-mark {
    width: 36px;
    height: 36px;
    background: #76b900;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 800;
    color: #000;
    letter-spacing: -1px;
    flex-shrink: 0;
}
.header-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.header-sub {
    font-size: 0.7rem;
    color: #6e8090;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1px;
}
.header-badge {
    font-size: 0.68rem;
    font-weight: 600;
    color: #76b900;
    background: rgba(118,185,0,0.1);
    border: 1px solid rgba(118,185,0,0.3);
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ---- Section labels ---- */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #5a6a7a;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2a3a;
}

/* ---- Status indicators ---- */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.73rem;
    font-weight: 600;
    color: #76b900;
}
.status-offline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.73rem;
    font-weight: 600;
    color: #e8451a;
}
.dot-online { width:7px; height:7px; border-radius:50%; background:#76b900; display:inline-block; }
.dot-offline { width:7px; height:7px; border-radius:50%; background:#e8451a; display:inline-block; }

/* ---- Sensitivity badge colours ---- */
.sl-public    { color:#4ade80; background:rgba(74,222,128,0.12); border:1px solid rgba(74,222,128,0.35); }
.sl-internal  { color:#60a5fa; background:rgba(96,165,250,0.12); border:1px solid rgba(96,165,250,0.35); }
.sl-confidential { color:#fde047; background:rgba(253,224,71,0.12); border:1px solid rgba(253,224,71,0.35); }
.sl-restricted{ color:#fb7185; background:rgba(251,113,133,0.12); border:1px solid rgba(251,113,133,0.35); }
.sl-badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.22rem 0.7rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: inline-block;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2a3a !important;
}
/* All normal text in sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p { color: #c9d1d9 !important; }
/* Bold section headers */
[data-testid="stSidebar"] .stMarkdown strong { color: #e6edf3 !important; font-size: 0.72rem; letter-spacing: 0.12em; }
/* h3 */
[data-testid="stSidebar"] h3 { color: #e6edf3 !important; }
/* Caption / small text */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p { color: #6e7f8d !important; font-size: 0.72rem !important; }
/* Toggle label */
[data-testid="stSidebar"] [data-testid="stToggle"] label { color: #c9d1d9 !important; }
/* Select/input labels */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #c9d1d9 !important; }

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    background: #0d1117;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 0.6rem;
    margin-bottom: 0.5rem;
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] textarea {
    background: #0d1117 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 2px rgba(118,185,0,0.15) !important;
}

/* ---- Buttons ---- */
.stButton button {
    background: #76b900 !important;
    color: #000 !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: background 0.2s !important;
}
.stButton button:hover { background: #8fcf00 !important; }

/* ---- Divider ---- */
hr { border-color: #1e2a3a !important; }

/* ---- Expander ---- */
[data-testid="stExpander"] {
    background: #0d1117;
    border: 1px solid #1e2a3a !important;
    border-radius: 6px !important;
}

/* ---- Uploader ---- */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: 1.5px dashed #1e2a3a !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}

/* ---- Alerts ---- */
[data-testid="stAlert"] { border-radius: 6px !important; }

/* ---- Select / Input ---- */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 5px !important;
    color: #c9d1d9 !important;
}

/* ---- Toggle ---- */
[data-testid="stToggle"] { accent-color: #76b900; }

scrollbar-width: thin;
scrollbar-color: #1e2a3a #080c14;
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def clear_chat_history():
    st.session_state.messages = []



def build_rag(file_paths: list, local_model: str = "gemma3:1b", chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Builds the RAG pipeline processing multiple files and creating vectorstore and chains.
    """
    # 0. Define Embeddings Builder (We need it for both loading and creating)
    # Use the full HF repo ID so the cached model is found and network
    # traffic is avoided once the model has been downloaded once.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    hasher = hashlib.md5()
    for fp in file_paths:
        with open(fp, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    file_hash = hasher.hexdigest()
    
    # Add chunk sizing to hash so if we change settings later, it makes a new DB
    db_id = f"{file_hash}_{chunk_size}_{chunk_overlap}_Offline"
    persist_dir = os.path.join(".faiss_storage", db_id)

    # 1 & 2 & 3. Load / Process Vector Store
    if os.path.exists(persist_dir):
        # Load from disk
        st.toast("Loading embeddings from cache.")
        vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        # 1. Load Documents
        st.toast("Generating embeddings for new documents...")
        docs = []
        for fp in file_paths:
            ext = os.path.splitext(fp)[-1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(fp)
            elif ext == '.txt':
                loader = TextLoader(fp, encoding="utf-8")
            elif ext in ['.doc', '.docx']:
                loader = Docx2txtLoader(fp)
            else:
                continue
            docs.extend(loader.load())

        # 2. Split Documents into smaller Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)

        # 3. Create Vector Store with Embeddings (Batching for Progress Bar)
        progress_text = "Embedding text chunks... Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        batch_size = 8 # Smaller batch size updates the progress bar more frequently and prevents server timeouts
        vectorstore = None
        total_splits = len(splits)
        
        for i in range(0, total_splits, batch_size):
            batch = splits[i:i + batch_size]
            
            # Retry mechanism for server disconnects
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
                    else:
                        vectorstore.add_documents(documents=batch)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2)
            
            # Update progress bar
            progress = min(1.0, (i + len(batch)) / total_splits)
            my_bar.progress(progress, text=f"Embedded {min(i + len(batch), total_splits)} / {total_splits} chunks...")
            
            # Brief pause to let the server breathe
            time.sleep(0.1)
            
        my_bar.empty()
        
        # Save to disk
        vectorstore.save_local(persist_dir)

    # 4. Set up Retriever and Language Model
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=local_model) 
        
    # Expose the LLM for guardrails to use
    st.session_state.llm_instance = llm 

    # 5. Create History-Aware Retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 6. Create RAG Chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer as concise as possible based on the context.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# ============================================================
# TIERED SAFETY SYSTEM
# ============================================================

JAILBREAK_PATTERNS = [
    "ignore previous", "forget your instructions", "ignore all prior",
    "jailbreak", "dan mode", "pretend you are", "act as if you are",
    "you are now", "disregard your", "override your"
]

SENSITIVITY_PROFILES = {
    "Public": {
        "description": "No data classification restrictions. Basic jailbreak protection only.",
        "input_patterns": [],
        "output_patterns": [],
        "badge_class": "sl-public",
    },
    "Internal": {
        "description": "Suitable for internal business data. Blocks credential and API key exposure.",
        "input_patterns": ["api key", "api_key", "password", "secret key", "access token", "private key", "credential"],
        "output_patterns": ["api_key", "api key", "password", "access_token", "credential", "private_key", "bearer token"],
        "badge_class": "sl-internal",
    },
    "Confidential": {
        "description": "For confidential data. Adds PII protection (emails, phone numbers, SSNs).",
        "input_patterns": [
            "api key", "api_key", "password", "secret key", "access token", "private key", "credential",
            "social security", "ssn", "date of birth", "home address", "phone number", "email address",
            "credit card", "bank account",
        ],
        "output_patterns": [
            "api_key", "api key", "password", "access_token", "credential", "private_key", "bearer token",
            "ssn", "social security", "date of birth", "credit card", "bank account",
        ],
        "badge_class": "sl-confidential",
    },
    "Restricted": {
        "description": "Maximum protection. For highly sensitive or regulated data (HIPAA, GDPR, financial).",
        "input_patterns": [
            "api key", "api_key", "password", "secret key", "access token", "private key", "credential",
            "social security", "ssn", "date of birth", "home address", "phone number", "email address",
            "credit card", "bank account", "medical record", "diagnosis", "prescription", "patient",
            "salary", "tax return", "financial statement", "trading", "investment",
        ],
        "output_patterns": [
            "api_key", "api key", "password", "access_token", "credential", "private_key", "bearer token",
            "ssn", "social security", "date of birth", "credit card", "bank account",
            "medical record", "diagnosis", "prescription", "patient id",
            "salary", "tax", "financial",
        ],
        "badge_class": "sl-restricted",
    },
}

def check_input_safety(user_input: str, sensitivity_level: str, guardrails_on: bool) -> str | None:
    if not guardrails_on:
        return None
    lower = user_input.lower()
    for pat in JAILBREAK_PATTERNS:
        if pat in lower:
            return "This request has been blocked. Prompt injection and instruction-override attempts are not permitted."
    profile = SENSITIVITY_PROFILES.get(sensitivity_level, SENSITIVITY_PROFILES["Internal"])
    for pat in profile["input_patterns"]:
        if pat in lower:
            return f"This request has been blocked under the active **{sensitivity_level}** data sensitivity policy."
    return None

def check_output_safety(response: str, sensitivity_level: str, guardrails_on: bool) -> str | None:
    if not guardrails_on:
        return None
    lower = response.lower()
    profile = SENSITIVITY_PROFILES.get(sensitivity_level, SENSITIVITY_PROFILES["Internal"])
    for pat in profile["output_patterns"]:
        if pat in lower:
            return f"[REDACTED — Output blocked by {sensitivity_level} data sensitivity policy.]"
    return None


# ============================================================
# TOP HEADER
# ============================================================
st.markdown("""
<div class="top-header">
  <div class="top-header-left">
    <div class="logo-mark">NV</div>
    <div>
      <div class="header-title">GUARDRAILS LOCAL RAG BOT</div>
      <div class="header-sub">Powered by NVIDIA NeMo Guardrails &nbsp;&bull;&nbsp; 100% Offline &nbsp;&bull;&nbsp; Local LLM</div>
    </div>
  </div>
  <div class="header-badge">OFFLINE &nbsp;/&nbsp; SECURE</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DOCUMENT UPLOAD (top of main area)
# ============================================================
st.markdown('<div class="section-label">Document Upload</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    label="Upload files — PDF, TXT or DOCX",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    label_visibility="visible"
)
if uploaded_files:
    names = "  |  ".join([f.name for f in uploaded_files])
    st.success(f"Loaded: {names}")

st.divider()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("**GUARDRAILS LOCAL RAG BOT**")
    st.caption("NVIDIA NeMo Guardrails — Local Inference")
    st.divider()

    st.markdown("**INFERENCE ENGINE**")
    ollama_running = is_ollama_running()
    if ollama_running:
        st.markdown('<span class="status-online"><span class="dot-online"></span> Ollama — Running</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline"><span class="dot-offline"></span> Ollama — Stopped</span>', unsafe_allow_html=True)
        if st.button("Start Ollama Server", use_container_width=True):
            with st.spinner("Starting Ollama server..."):
                if start_ollama():
                    st.success("Ollama is now running.")
                    st.rerun()
                else:
                    st.error("Failed to start Ollama. Verify it is installed and in your system PATH.")

    st.markdown("")
    if ollama_running:
        installed_models = get_installed_models()
        if installed_models:
            default_index = installed_models.index("gemma3:1b") if "gemma3:1b" in installed_models else 0
            local_model = st.selectbox("Active Model", options=installed_models, index=default_index)
        else:
            local_model = st.text_input("Model Identifier", value="gemma3:1b",
                                        help="No local models detected. Run: ollama pull <model>")
    else:
        local_model = st.text_input("Model Identifier", value="gemma3:1b",
                                    help="Start the Ollama server to browse installed models.")
    st.divider()

    st.markdown("**SAFETY & PRIVACY CONTROLS**")
    enable_guardrails = st.toggle("Enable Guardrails", value=True,
                                  help="Master switch for all NeMo Guardrails safety filters.")

    sensitivity_level = st.selectbox(
        "Data Sensitivity Level",
        options=list(SENSITIVITY_PROFILES.keys()),
        index=1,
        help="Controls how aggressively the system filters inputs and outputs based on data classification.",
        disabled=not enable_guardrails
    )

    profile = SENSITIVITY_PROFILES[sensitivity_level]
    badge_cls = profile["badge_class"]
    st.markdown(
        f'<span class="sl-badge {badge_cls}">{sensitivity_level}</span>'
        f'<br><small style="color:#5a6a7a; font-size:0.72rem;">{profile["description"]}</small>',
        unsafe_allow_html=True
    )

    with st.expander("Sensitivity Level Reference"):
        st.markdown("""
| Level | Protection |
|---|---|
| **Public** | Jailbreak / prompt injection only |
| **Internal** | + API keys, credentials, passwords |
| **Confidential** | + PII (SSN, email, phone, DOB, credit card) |
| **Restricted** | + Medical records, financial, HIPAA/GDPR |
        """)

    st.divider()

    st.markdown("**ADVANCED**")
    with st.expander("Chunking Parameters"):
        chunk_size = st.number_input("Chunk Size (chars)", value=1000, step=100)
        chunk_overlap = st.number_input("Chunk Overlap (chars)", value=200, step=50)

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        clear_chat_history()
        st.rerun()

    st.markdown("")
    st.caption("GUARDRAILS LOCAL RAG BOT v1.0 — Offline · Open Source")
    st.markdown(
        '<a href="https://github.com/sowmiyan-s" target="_blank" '
        'style="font-size:0.72rem; color:#6e7f8d; text-decoration:none; '
        'letter-spacing:0.04em; display:block; margin-top:0.25rem;">'
        '&#128279; github.com/sowmiyan-s</a>',
        unsafe_allow_html=True
    )

# ============================================================
# PROCESS UPLOADED DOCUMENTS
# ============================================================
if not uploaded_files:
    st.session_state.rag_chain = None
    st.session_state.current_file_names = []

elif uploaded_files:
    current_file_names = [file.name for file in uploaded_files]
    config_mtime = os.path.getmtime('./guardrails_config/config.yml')
    needs_rebuild = (
        not st.session_state.rag_chain or
        st.session_state.get("current_file_names") != current_file_names or
        st.session_state.get("local_model") != local_model or
        st.session_state.get("chunk_size") != chunk_size or
        st.session_state.get("chunk_overlap") != chunk_overlap or
        st.session_state.get("enable_guardrails") != enable_guardrails or
        st.session_state.get("sensitivity_level") != sensitivity_level or
        st.session_state.get("config_mtime") != config_mtime
    )

    if needs_rebuild:
        with st.spinner("Embedding documents offline... Please wait."):
            try:
                temp_file_paths = []
                for uploaded_file in uploaded_files:
                    ext = "." + uploaded_file.name.split('.')[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_paths.append(temp_file.name)

                raw_chain = build_rag(temp_file_paths, local_model, chunk_size, chunk_overlap)
                st.session_state.rag_chain = raw_chain

                st.session_state.current_file_names = current_file_names
                st.session_state.local_model = local_model
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap
                st.session_state.enable_guardrails = enable_guardrails
                st.session_state.sensitivity_level = sensitivity_level
                st.session_state.config_mtime = config_mtime

                for path in temp_file_paths:
                    os.remove(path)

                st.session_state.messages = []
                st.toast("Documents ready. You may now begin querying.")
            except Exception as e:
                st.error(f"Error processing Documents: {e}")



# ============================================================
# CHAT AREA
# ============================================================
if not st.session_state.rag_chain:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem;">
        <p style="font-size: 1rem; font-weight: 500; color: #4a6070; margin-top: 1rem; letter-spacing: 0.02em;">
            Upload one or more documents above to begin.
        </p>
        <p style="font-size: 0.78rem; color: #3a5060; letter-spacing: 0.06em; text-transform: uppercase;">
            Supported: PDF &nbsp;&bull;&nbsp; TXT &nbsp;&bull;&nbsp; DOCX
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 1rem;">
            <p style="font-size:0.9rem; font-weight:500; letter-spacing:0.04em; color:#4a6070;">Documents indexed. You may now query the knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Enter your question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            message_placeholder = st.empty()
            full_response = ""

            try:
                enable_guardrails = st.session_state.get("enable_guardrails", True)
                sensitivity_level = st.session_state.get("sensitivity_level", "Internal")

                # Step 1: Input safety check
                blocked = check_input_safety(prompt, sensitivity_level, enable_guardrails)
                if blocked:
                    message_placeholder.markdown(blocked)
                    st.session_state.messages.append({"role": "assistant", "content": blocked})
                    st.stop()

                # Step 2: Run RAG chain
                with st.spinner("Processing..."):
                    result = st.session_state.rag_chain.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })

                # Extract answer
                if isinstance(result, dict) and "answer" in result:
                    full_response = result["answer"]
                elif isinstance(result, dict) and "output" in result:
                    full_response = result["output"]
                elif isinstance(result, str):
                    full_response = result
                else:
                    full_response = str(result)

                # Step 3: Output safety check
                blocked_out = check_output_safety(full_response, sensitivity_level, enable_guardrails)
                if blocked_out:
                    full_response = blocked_out

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")

/* ═══════════════════════════════════════════════════════════════════════════
   GUARDRAILS LOCAL RAG BOT — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */
'use strict';

const API_BASE = window.location.origin;

// ─── Default Ollama endpoint ──────────────────────────────────────────────────
const DEFAULT_OLLAMA_ENDPOINT = 'http://localhost:11434';

// ─── App state ────────────────────────────────────────────────────────────────
const state = {
  sessionId: null,
  ollamaRunning: false,
  models: [],
  selectedFiles: [],
  isProcessing: false,
  isChatting: false,
  uploadOpen: true,
  currentDbId: null,
  serverConfig: null,
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const sidebar = $('sidebar');
const sidebarBackdrop = $('sidebarBackdrop');
const sidebarOpen = $('sidebarOpen');
const sidebarClose = $('sidebarClose');
const sidebarCollapseBtn = $('sidebarCollapseBtn');
const sidebarExpandBtn = $('sidebarExpandBtn');
const sidebarBody = $('sidebarBody');
const sidebarRail = $('sidebarRail');
const railStatusDot = $('railStatusDot');
const railClearBtn = $('railClearBtn');

const ollamaEndpointInput = $('ollamaEndpoint');
const ollamaStatus = $('ollamaStatus');
const ollamaStatusText = $('ollamaStatusText');
const btnStartOllama = $('btnStartOllama');
const modelSelect = $('modelSelect');
const connectionBadge = $('connectionBadge');

const guardrailsToggle = $('guardrailsToggle');
const sensitivitySelect = $('sensitivitySelect');
const sensitivityBadge = $('sensitivityBadge');
const sensitivityBadgeLabel = $('sensitivityBadgeLabel');
const sensitivityDesc = $('sensitivityDesc');
const sensitivityHint = $('sensitivityHint');
const guardrailsIndicator = $('guardrailsIndicator');

const chunkSize = $('chunkSize');
const chunkOverlap = $('chunkOverlap');

const storagePool = $('storagePool');
const storageEmpty = $('storageEmpty');
const btnRefreshStorage = $('btnRefreshStorage');

const uploadPanelToggle = $('uploadPanelToggle');
const uploadSection = $('uploadSection');
const uploadToggleIcon = $('uploadToggleIcon');
const uploadFileCount = $('uploadFileCount');
const uploadSectionBody = $('uploadSectionBody');
const dropZone = $('dropZone');
const dropClick = $('dropClick');
const fileInput = $('fileInput');
const fileList = $('fileList');
const btnProcess = $('btnProcess');
const btnReset = $('btnReset');
const uploadStatus = $('uploadStatus');

const emptyState = $('emptyState');
const chatMessages = $('chatMessages');
const chatSection = $('chatSection');
const readyBanner = $('readyBanner');
const inputBarWrapper = $('inputBarWrapper');
const typingIndicator = $('typingIndicator');
const chatInput = $('chatInput');
const btnSend = $('btnSend');
const btnClear = $('btnClear');
const toastContainer = $('toastContainer');

const tunnelModal = $('tunnelModal');
const tunnelModalClose = $('tunnelModalClose');

// ─── Ollama endpoint (stored in localStorage) ─────────────────────────────────
function getOllamaEndpoint() {
  return (ollamaEndpointInput.value || '').trim() || DEFAULT_OLLAMA_ENDPOINT;
}

function loadSavedEndpoint() {
  const saved = localStorage.getItem('ragbot_ollama_endpoint');
  if (saved) ollamaEndpointInput.value = saved;
  else ollamaEndpointInput.value = DEFAULT_OLLAMA_ENDPOINT;
}

ollamaEndpointInput.addEventListener('change', () => {
  const val = ollamaEndpointInput.value.trim();
  localStorage.setItem('ragbot_ollama_endpoint', val || DEFAULT_OLLAMA_ENDPOINT);
  // Re-check health immediately when endpoint changes
  refreshHealth();
});

// Show tunnel modal when endpoint looks remote (not localhost)
ollamaEndpointInput.addEventListener('blur', () => {
  const val = (ollamaEndpointInput.value || '').trim();
  const isLocal = !val || val.includes('localhost') || val.includes('127.0.0.1');
  if (!isLocal && val) {
    // No-op: just silently accept remote URLs. Modal is shown from the hint link.
  }
});

// ─── Utility ──────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(API_BASE + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function toast(msg, type = 'info', duration = 3400) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut 0.22s ease forwards';
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, duration);
}

// ─── Tunnel modal ─────────────────────────────────────────────────────────────
tunnelModalClose.addEventListener('click', () => { tunnelModal.style.display = 'none'; });
tunnelModal.addEventListener('click', e => { if (e.target === tunnelModal) tunnelModal.style.display = 'none'; });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && tunnelModal.style.display !== 'none') tunnelModal.style.display = 'none';
});

// ─── SIDEBAR — Mobile drawer ───────────────────────────────────────────────────
function openMobileSidebar() {
  sidebar.classList.add('open');
  sidebarBackdrop.classList.add('visible');
  sidebarOpen.setAttribute('aria-expanded', 'true');
  setTimeout(() => sidebarClose.focus(), 50);
}
function closeMobileSidebar() {
  sidebar.classList.remove('open');
  sidebarBackdrop.classList.remove('visible');
  sidebarOpen.setAttribute('aria-expanded', 'false');
}

sidebarOpen.addEventListener('click', openMobileSidebar);
sidebarClose.addEventListener('click', closeMobileSidebar);
sidebarBackdrop.addEventListener('click', closeMobileSidebar);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && sidebar.classList.contains('open')) closeMobileSidebar();
});

// ─── SIDEBAR — Desktop collapse / expand ──────────────────────────────────────
function collapseSidebar() {
  state.sidebarCollapsed = true;
  sidebar.classList.add('collapsed');
  sidebarExpandBtn.style.display = 'flex';
  sidebarCollapseBtn.setAttribute('aria-label', 'Expand sidebar');
  syncRailStatus();
}

function expandSidebar() {
  state.sidebarCollapsed = false;
  sidebar.classList.remove('collapsed');
  sidebarExpandBtn.style.display = 'none';
  sidebarCollapseBtn.setAttribute('aria-label', 'Collapse sidebar');
}

if (sidebarCollapseBtn) sidebarCollapseBtn.addEventListener('click', collapseSidebar);
if (sidebarExpandBtn) sidebarExpandBtn.addEventListener('click', expandSidebar);

if (railClearBtn) { railClearBtn.addEventListener('click', clearConversation); }

function syncRailStatus() {
  if (!railStatusDot) return;
  railStatusDot.className = 'status-dot';
  if (state.ollamaRunning) {
    railStatusDot.style.background = 'var(--accent)';
    railStatusDot.style.boxShadow = '0 0 5px var(--accent)';
  } else {
    railStatusDot.style.background = 'var(--danger)';
    railStatusDot.style.boxShadow = '0 0 5px var(--danger)';
  }
}

// ─── UPLOAD PANEL — Toggle collapse ───────────────────────────────────────────
function setUploadPanelOpen(open) {
  state.uploadOpen = open;
  if (open) {
    uploadSection.classList.remove('collapsed');
    uploadPanelToggle.classList.remove('collapsed');
    uploadPanelToggle.setAttribute('aria-expanded', 'true');
  } else {
    uploadSection.classList.add('collapsed');
    uploadPanelToggle.classList.add('collapsed');
    uploadPanelToggle.setAttribute('aria-expanded', 'false');
  }
}

uploadPanelToggle.addEventListener('click', () => setUploadPanelOpen(!state.uploadOpen));

function autoCollapseUpload() {
  if (state.uploadOpen) setUploadPanelOpen(false);
}

// ─── OLLAMA STATUS ────────────────────────────────────────────────────────────
async function refreshHealth() {
  const endpoint = getOllamaEndpoint();
  const params = new URLSearchParams({ ollama_host: endpoint });
  try {
    const data = await apiFetch(`/api/health?${params}`);
    state.ollamaRunning = data.ollama_running;
    state.models = data.models || [];

    if (data.ollama_running) {
      ollamaStatus.className = 'status-badge status-online';
      const isLocal = endpoint.includes('localhost') || endpoint.includes('127.0.0.1');
      ollamaStatusText.textContent = isLocal ? 'Ollama — Running (local)' : 'Ollama — Connected (remote)';
      btnStartOllama.style.display = 'none';
      connectionBadge.textContent = isLocal ? 'OFFLINE / SECURE' : 'TUNNELLED / SECURE';
      connectionBadge.style.color = isLocal ? 'var(--accent)' : 'var(--info)';
      connectionBadge.style.borderColor = isLocal ? 'var(--accent-border)' : 'rgba(96,165,250,0.35)';
      connectionBadge.style.background = isLocal ? 'var(--accent-dim)' : 'rgba(96,165,250,0.10)';

      modelSelect.innerHTML = '';
      const models = state.models.length ? state.models : ['gemma3:1b'];
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        if (m === 'gemma3:1b') opt.selected = true;
        modelSelect.appendChild(opt);
      });
    } else {
      ollamaStatus.className = 'status-badge status-offline';
      const isLocal = endpoint.includes('localhost') || endpoint.includes('127.0.0.1');
      ollamaStatusText.textContent = isLocal ? 'Ollama — Stopped' : 'Ollama — Unreachable';
      btnStartOllama.style.display = isLocal ? 'block' : 'none';
      connectionBadge.textContent = 'OFFLINE / SECURE';
      connectionBadge.style.color = '';
      connectionBadge.style.borderColor = '';
      connectionBadge.style.background = '';
    }
  } catch {
    ollamaStatus.className = 'status-badge status-offline';
    ollamaStatusText.textContent = 'Ollama — Unreachable';
    btnStartOllama.style.display = 'block';
  }
  syncRailStatus();
}

btnStartOllama.addEventListener('click', async () => {
  btnStartOllama.disabled = true;
  btnStartOllama.textContent = '⏳ Starting…';
  try {
    await apiFetch('/api/ollama/start', { method: 'POST' });
    toast('Ollama started!', 'success');
    await refreshHealth();
  } catch (e) {
    toast(e.message, 'error', 5000);
  } finally {
    btnStartOllama.disabled = false;
    btnStartOllama.textContent = '▶ Start Ollama';
  }
});

// ─── SENSITIVITY / GUARDRAILS ─────────────────────────────────────────────────
const SENSITIVITY_META = {
  Public: { badge: 'badge-public', hint: 'No extra filters — jailbreak protection only.' },
  Internal: { badge: 'badge-internal', hint: '🛡 <strong>Internal</strong> — API keys &amp; credentials protected.' },
  Confidential: { badge: 'badge-confidential', hint: '🛡 <strong>Confidential</strong> — PII (SSN, email, phone) protected.' },
  Restricted: { badge: 'badge-restricted', hint: '🛡 <strong>Restricted</strong> — Medical, financial, HIPAA/GDPR protected.' },
};
const SENSITIVITY_DESC = {
  Public: 'No data classification restrictions. Basic jailbreak protection only.',
  Internal: 'Suitable for internal business data. Blocks credential and API key exposure.',
  Confidential: 'For confidential data. Adds PII protection (emails, phone, SSNs).',
  Restricted: 'Maximum protection for HIPAA/GDPR/financial data.',
};

function updateSensitivityUI() {
  const level = sensitivitySelect.value;
  const meta = SENSITIVITY_META[level];
  sensitivityBadge.className = `sensitivity-badge ${meta.badge}`;
  sensitivityBadgeLabel.textContent = level.toUpperCase();
  sensitivityDesc.textContent = SENSITIVITY_DESC[level];
  sensitivityHint.innerHTML = guardrailsToggle.checked ? meta.hint : '⚪ Guardrails disabled.';
}

sensitivitySelect.addEventListener('change', updateSensitivityUI);
guardrailsToggle.addEventListener('change', () => {
  sensitivitySelect.disabled = !guardrailsToggle.checked;
  guardrailsIndicator.classList.toggle('active', guardrailsToggle.checked);
  updateSensitivityUI();
});

// ─── STORAGE POOL (Document Library) ─────────────────────────────────────────
async function loadStoragePool() {
  storageEmpty.textContent = 'Loading…';
  storageEmpty.style.display = 'block';
  // Remove old collection cards
  storagePool.querySelectorAll('.storage-card').forEach(c => c.remove());

  try {
    const data = await apiFetch('/api/storage');
    const collections = data.collections || [];

    if (collections.length === 0) {
      storageEmpty.textContent = 'No indexed collections yet.';
      return;
    }

    storageEmpty.style.display = 'none';

    collections.forEach(col => {
      const card = document.createElement('div');
      card.className = 'storage-card' + (col.available ? '' : ' unavailable');
      card.dataset.dbId = col.db_id;

      const names = col.files.join(', ') || 'Unknown files';
      const date = col.created_at ? new Date(col.created_at).toLocaleDateString() : '—';

      card.innerHTML = `
        <div class="storage-card-header">
          <span class="storage-card-name" title="${escapeHtml(names)}">${escapeHtml(truncate(names, 34))}</span>
          <button class="storage-del-btn" data-db-id="${col.db_id}" title="Delete this collection" aria-label="Delete">✕</button>
        </div>
        <div class="storage-card-meta">
          <span class="storage-meta-tag">${col.model || '?'}</span>
          <span class="storage-meta-tag">${date}</span>
          ${!col.available ? '<span class="storage-meta-tag unavail">⚠ Missing</span>' : ''}
        </div>
        <button class="btn btn-secondary storage-load-btn" data-db-id="${col.db_id}" ${!col.available ? 'disabled' : ''}>
          ↩ Load Session
        </button>
      `;

      card.querySelector('.storage-load-btn').addEventListener('click', () => loadStoredSession(col));
      card.querySelector('.storage-del-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteStoredSession(col.db_id, card);
      });

      storagePool.appendChild(card);
    });
  } catch (e) {
    storageEmpty.textContent = `Error loading library: ${e.message}`;
  }
}

async function loadStoredSession(col) {
  if (!state.ollamaRunning) {
    toast('Ollama is not connected. Set the endpoint in the sidebar.', 'error');
    return;
  }

  // Highlight active card
  storagePool.querySelectorAll('.storage-card').forEach(c => c.classList.remove('active'));
  const card = storagePool.querySelector(`[data-db-id="${col.db_id}"]`);
  if (card) {
    card.classList.add('active');
    card.querySelector('.storage-load-btn').textContent = '⏳ Loading…';
    card.querySelector('.storage-load-btn').disabled = true;
  }

  try {
    const data = await apiFetch('/api/sessions/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        db_id: col.db_id,
        model: modelSelect.value,
        ollama_host: getOllamaEndpoint(),
      }),
    });

    state.sessionId = data.session_id;
    state.currentDbId = col.db_id;

    // Update upload status text
    setUploadStatus(`✓ Loaded: ${data.files.join(', ')}`, 'success');
    toast(`Loaded "${data.files.join(', ')}" from library.`, 'success');
    showChatReady();
    autoCollapseUpload();
    if (window.innerWidth <= 900) closeMobileSidebar();
  } catch (e) {
    toast(`Failed to load: ${e.message}`, 'error', 6000);
    if (card) {
      card.classList.remove('active');
      card.querySelector('.storage-load-btn').textContent = '↩ Load Session';
      card.querySelector('.storage-load-btn').disabled = false;
    }
  }
}

async function deleteStoredSession(dbId, cardEl) {
  if (!confirm('Delete this indexed collection from disk? This cannot be undone.')) return;
  try {
    await apiFetch('/api/storage/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ db_id: dbId }),
    });
    cardEl.style.animation = 'toastOut 0.2s ease forwards';
    setTimeout(() => { cardEl.remove(); checkStorageEmpty(); }, 220);
    toast('Collection deleted.', 'info');
  } catch (e) {
    toast(`Delete failed: ${e.message}`, 'error');
  }
}

function checkStorageEmpty() {
  if (!storagePool.querySelector('.storage-card')) {
    storageEmpty.textContent = 'No indexed collections yet.';
    storageEmpty.style.display = 'block';
  }
}

btnRefreshStorage.addEventListener('click', loadStoragePool);

// ─── FILE HANDLING ────────────────────────────────────────────────────────────
const getExt = name => name.split('.').pop().toLowerCase();
const truncate = (s, n) => s.length > n ? s.slice(0, n) + '…' : s;

function renderFileList() {
  const n = state.selectedFiles.length;
  if (n === 0) {
    fileList.style.display = 'none';
    btnProcess.disabled = true;
    btnReset.style.display = 'none';
    uploadFileCount.style.display = 'none';
    return;
  }
  fileList.style.display = 'flex';
  btnProcess.disabled = false;
  btnReset.style.display = 'inline-flex';
  uploadFileCount.textContent = `${n} file${n > 1 ? 's' : ''}`;
  uploadFileCount.style.display = 'inline-flex';

  fileList.innerHTML = '';
  state.selectedFiles.forEach(f => {
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    chip.setAttribute('role', 'listitem');
    chip.innerHTML = `<span class="ext-tag">${getExt(f.name)}</span>${f.name}`;
    fileList.appendChild(chip);
  });
}

function addFiles(files) {
  const allowed = new Set(['pdf', 'txt', 'doc', 'docx']);
  const incoming = Array.from(files);
  const valid = incoming.filter(f => allowed.has(getExt(f.name)));
  if (valid.length < incoming.length) toast('Some files skipped (unsupported format).', 'warn');
  const names = new Set(state.selectedFiles.map(f => f.name));
  valid.filter(f => !names.has(f.name)).forEach(f => state.selectedFiles.push(f));
  renderFileList();
}

dropZone.addEventListener('click', () => fileInput.click());
dropClick.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', e => {
  if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });

btnReset.addEventListener('click', () => {
  state.selectedFiles = [];
  state.sessionId = null;
  state.currentDbId = null;
  renderFileList();
  setUploadStatus('');
  setUploadPanelOpen(true);
  showEmptyState();
});

// ─── UPLOAD / PROCESS ─────────────────────────────────────────────────────────
function setUploadStatus(msg, type = '') {
  uploadStatus.textContent = msg;
  uploadStatus.className = `upload-status ${type}`.trim();
}

btnProcess.addEventListener('click', processDocuments);

async function processDocuments() {
  if (state.isProcessing || !state.selectedFiles.length) return;
  if (!state.ollamaRunning) {
    toast('Ollama is not running. Set the Ollama Endpoint in the sidebar.', 'error');
    return;
  }

  state.isProcessing = true;
  btnProcess.disabled = true;
  btnReset.disabled = true;
  setUploadStatus('⏳ Embedding documents — please wait…', 'loading');

  const form = new FormData();
  state.selectedFiles.forEach(f => form.append('files', f));
  form.append('model', modelSelect.value);
  form.append('chunk_size', chunkSize.value);
  form.append('chunk_overlap', chunkOverlap.value);
  form.append('ollama_host', getOllamaEndpoint());

  try {
    const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    state.sessionId = data.session_id;
    state.currentDbId = data.db_id;
    setUploadStatus(`✓ ${data.files.length} document(s) indexed.`, 'success');
    toast('Documents ready — start chatting!', 'success');
    showChatReady();
    autoCollapseUpload();
    // Refresh the library so the new collection appears
    await loadStoragePool();
  } catch (e) {
    setUploadStatus(`✗ ${e.message}`, 'error');
    toast(e.message, 'error', 6000);
  } finally {
    state.isProcessing = false;
    btnProcess.disabled = false;
    btnReset.disabled = false;
  }
}

// ─── CHAT STATE TRANSITIONS ───────────────────────────────────────────────────
function showEmptyState() {
  emptyState.style.display = '';
  chatMessages.style.display = 'none';
  readyBanner.style.display = 'none';
  typingIndicator.style.display = 'none';
  inputBarWrapper.style.display = 'none';
  chatInput.disabled = true;
  btnSend.disabled = true;
  chatMessages.innerHTML = '';
}

function showChatReady() {
  emptyState.style.display = 'none';
  chatMessages.style.display = 'flex';
  readyBanner.style.display = 'flex';
  inputBarWrapper.style.display = 'block';
  chatInput.disabled = false;
  btnSend.disabled = false;
  requestAnimationFrame(() => chatInput.focus());
}

// ─── CLEAR CONVERSATION ───────────────────────────────────────────────────────
async function clearConversation() {
  if (state.sessionId) {
    try {
      await apiFetch('/api/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
    } catch { /* best-effort */ }
  }
  chatMessages.innerHTML = '';
  if (state.sessionId) readyBanner.style.display = 'flex';
  toast('Conversation cleared.', 'info');
  if (window.innerWidth <= 900) closeMobileSidebar();
}

btnClear.addEventListener('click', clearConversation);

// ─── CHAT INPUT ────────────────────────────────────────────────────────────────
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + 'px';
  btnSend.disabled = !chatInput.value.trim() || state.isChatting;
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
btnSend.addEventListener('click', sendMessage);

// ─── SEND MESSAGE ─────────────────────────────────────────────────────────────
async function sendMessage() {
  const question = chatInput.value.trim();
  if (!question || state.isChatting || !state.sessionId) return;

  state.isChatting = true;
  chatInput.value = '';
  chatInput.style.height = 'auto';
  btnSend.disabled = true;
  chatInput.disabled = true;
  readyBanner.style.display = 'none';

  appendMessage('user', question);
  typingIndicator.style.display = 'flex';
  scrollToBottom();

  try {
    const data = await apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        question,
        model: modelSelect.value,
        enable_guardrails: guardrailsToggle.checked,
        sensitivity_level: sensitivitySelect.value,
        ollama_host: getOllamaEndpoint(),
      }),
    });
    typingIndicator.style.display = 'none';
    appendMessage('assistant', data.answer, data.blocked);
  } catch (e) {
    typingIndicator.style.display = 'none';
    appendMessage('assistant', `⚠️ Error: ${e.message}`, false, true);
    toast(e.message, 'error', 6000);
  } finally {
    state.isChatting = false;
    chatInput.disabled = false;
    btnSend.disabled = false;
    requestAnimationFrame(() => chatInput.focus());
  }
}

// ─── RENDER MESSAGE ───────────────────────────────────────────────────────────
function appendMessage(role, text, blocked = false, isError = false) {
  const wrap = document.createElement('div');
  wrap.className = `chat-message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `msg-avatar ${role === 'user' ? 'user-av' : 'bot-av'}`;
  avatar.setAttribute('aria-hidden', 'true');
  avatar.textContent = role === 'user' ? 'U' : 'NV';

  const bubble = document.createElement('div');
  bubble.className = `msg-bubble${blocked || isError ? ' blocked' : ''}`;

  if (blocked) {
    const label = document.createElement('div');
    label.className = 'guard-label';
    label.textContent = '🛡 GUARDRAIL TRIGGERED';
    bubble.appendChild(label);
  }

  const content = document.createElement('div');
  content.innerHTML = role === 'assistant'
    ? marked.parse(text)
    : escapeHtml(text);
  bubble.appendChild(content);

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  scrollToBottom();
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    if (chatSection) chatSection.scrollTop = chatSection.scrollHeight;
  });
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
(async function init() {
  await fetchConfig();
  updateSensitivityUI();

  state.sidebarCollapsed = false;
  sidebarExpandBtn.style.display = 'none';

  await refreshHealth();
  await loadStoragePool();

  setInterval(refreshHealth, 12_000);
})();

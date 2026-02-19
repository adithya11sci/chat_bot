/* ══════════════════════════════════════════════════════════════
   BookMind — app.js
   Flow:
     1. Page loads → check /dataset/status
        → ready?   → enable chat
        → not ready → chat disabled, must upload first
     2. Upload CSV → POST /upload → backend builds index
        → on success → enable chat
     3. Chat: POST /chat → display response + book cards
   ══════════════════════════════════════════════════════════════ */

/* ── State ─────────────────────────────────────────────────────────────────── */
let isLoading = false;
let isUploading = false;
let datasetReady = false;

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const booksListEl = document.getElementById('books-list');
const booksCount = document.getElementById('books-count');
const toastEl = document.getElementById('toast');
const chipsRow = document.getElementById('chips-row');

// Upload
const fileInput = document.getElementById('csv-file-input');
const uploadBtn = document.getElementById('upload-btn');
const uploadBtnTxt = document.getElementById('upload-btn-text');
const indexingBar = document.getElementById('indexing-bar');
const indexingFill = document.getElementById('indexing-fill');
const indexingLbl = document.getElementById('indexing-label');

// Header badge
const badgeDot = document.getElementById('badge-dot');
const badgeText = document.getElementById('badge-text');


/* ══════════════════════════════════════════════════════════════
   STARTUP
   ══════════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', async () => {
  setChatEnabled(false);
  setBadge('loading', 'Loading…');

  try {
    const res = await fetch('/dataset/status');
    const info = await res.json();
    if (info.ready) {
      onDatasetReady(info.name, info.rows);
      // Update welcome message for pre-loaded dataset
      appendMessage('bot', `📊 Dataset "${info.name}" is loaded and ready (${Number(info.rows).toLocaleString()} rows). Ask me anything!`);
    } else {
      setBadge('empty', 'No dataset loaded');
    }
  } catch {
    setBadge('error', 'Server error');
  }
});


/* ══════════════════════════════════════════════════════════════
   UPLOAD
   ══════════════════════════════════════════════════════════════ */
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) handleUpload(file);
  fileInput.value = '';
});

async function handleUpload(file) {
  if (isUploading) return;
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showToast('Only CSV files are supported.');
    return;
  }

  isUploading = true;
  setChatEnabled(false);
  datasetReady = false;

  // UI: show indexing state
  uploadBtn.classList.add('disabled');
  uploadBtnTxt.textContent = 'Processing…';
  setBadge('loading', `Indexing ${file.name}…`);
  showIndexingBar(`Uploading "${file.name}"…`, 5);

  appendMessage('bot', `📂 Uploading "${file.name}"… Building the search index, please wait.`);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const progressTimer = animateProgress(5, 90, 20000);

    const res = await fetch('/upload', { method: 'POST', body: formData });
    clearInterval(progressTimer);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();

    if (data.status === 'error') {
      throw new Error(data.message);
    }

    // Success
    setIndexingProgress(100, 'Done!');
    await sleep(500);
    hideIndexingBar();

    if (data.status === 'already_loaded') {
      showToast(`⚡ Already indexed — ${data.name}`);
      appendMessage('bot', `⚡ "${data.name}" is already indexed and ready! Go ahead and ask questions.`);
    } else {
      showToast(`✅ Indexed ${Number(data.rows).toLocaleString()} rows`);
      appendMessage('bot', `✅ "${data.name}" has been uploaded and indexed! (${Number(data.rows).toLocaleString()} rows). Ask me anything about this dataset.`);
    }

    onDatasetReady(data.name, data.rows);

  } catch (err) {
    hideIndexingBar();
    showToast(`Error: ${err.message}`);
    appendMessage('bot', `⚠️ Upload failed: ${err.message}`);
    setBadge('error', 'Upload failed');
    uploadBtn.classList.remove('disabled');
    uploadBtnTxt.textContent = 'Upload Dataset';
  } finally {
    isUploading = false;
  }
}


/* ══════════════════════════════════════════════════════════════
   STATE HELPERS
   ══════════════════════════════════════════════════════════════ */
function onDatasetReady(name, rows) {
  datasetReady = true;
  setChatEnabled(true);
  setBadge('ready', `${name} · ${Number(rows).toLocaleString()} rows`);
  uploadBtn.classList.remove('disabled');
  uploadBtnTxt.textContent = 'Change Dataset';

  // Update the welcome message to show dataset info
  const welcomeMsg = document.getElementById('welcome-msg');
  if (welcomeMsg) {
    const bubble = welcomeMsg.querySelector('.bubble');
    if (bubble) {
      bubble.innerHTML = `
        <p>Hello! I'm <strong>BookMind</strong>, your AI data assistant.</p>
        <p style="margin-top:8px">📊 Active dataset: <strong>${escapeHtml(name)}</strong> — <strong>${Number(rows).toLocaleString()}</strong> rows loaded.</p>
        <p style="margin-top:4px">Ask me anything about this data!</p>
      `;
    }
  }
}

function setChatEnabled(enabled) {
  inputEl.disabled = !enabled;
  sendBtn.disabled = !enabled;
  inputEl.placeholder = enabled
    ? 'Ask anything about your dataset…'
    : 'Upload a dataset first…';
}

function setBadge(state, text) {
  badgeDot.className = 'badge-dot';
  if (state === 'ready') badgeDot.classList.add('dot-ready');
  if (state === 'loading') badgeDot.classList.add('dot-loading');
  if (state === 'error') badgeDot.classList.add('dot-error');
  badgeText.textContent = text;
}

// Indexing progress bar
function showIndexingBar(label, pct) {
  indexingBar.style.display = 'block';
  indexingFill.style.width = pct + '%';
  indexingLbl.textContent = label;
}
function setIndexingProgress(pct, label) {
  indexingFill.style.width = pct + '%';
  indexingLbl.textContent = label;
}
function hideIndexingBar() {
  indexingBar.style.display = 'none';
  indexingFill.style.width = '0%';
}
function animateProgress(from, to, durationMs) {
  const steps = 60, stepTime = durationMs / steps, stepSize = (to - from) / steps;
  let cur = from;
  return setInterval(() => {
    cur = Math.min(cur + stepSize, to);
    indexingFill.style.width = cur + '%';
    indexingLbl.textContent = `Indexing… ${Math.round(cur)}%`;
  }, stepTime);
}


/* ══════════════════════════════════════════════════════════════
   CHAT
   ══════════════════════════════════════════════════════════════ */
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isLoading && datasetReady) sendMessage(e);
  }
});

function fillQuery(text) {
  if (!datasetReady) { showToast('Upload a dataset first!'); return; }
  inputEl.value = text;
  inputEl.focus();
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

async function sendMessage(e) {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question || isLoading || !datasetReady) return;

  chipsRow.style.display = 'none';
  appendMessage('user', question);
  inputEl.value = '';
  inputEl.style.height = 'auto';

  const typingId = showTyping();
  setLoading(true);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    removeTyping(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    appendMessage('bot', data.response);
    renderBooks(data.metadata || []);

  } catch (err) {
    removeTyping(typingId);
    appendMessage('bot', `⚠️ ${err.message}`);
    showToast(err.message);
  } finally {
    setLoading(false);
  }
}

function setLoading(val) {
  isLoading = val;
  sendBtn.disabled = val || !datasetReady;
  inputEl.disabled = val || !datasetReady;
}


/* ══════════════════════════════════════════════════════════════
   MESSAGES
   ══════════════════════════════════════════════════════════════ */
function appendMessage(role, text) {
  const isBot = role === 'bot';
  const div = document.createElement('div');
  div.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
  div.innerHTML = `
    <div class="avatar ${isBot ? 'bot-avatar' : 'user-avatar'}">${isBot ? '🤖' : '👤'}</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function showTyping() {
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.className = 'message bot-message';
  div.id = id;
  div.innerHTML = `
    <div class="avatar bot-avatar">🤖</div>
    <div class="bubble typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  messagesEl.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}


/* ══════════════════════════════════════════════════════════════
   BOOK CARDS
   ══════════════════════════════════════════════════════════════ */
function renderBooks(books) {
  booksListEl.innerHTML = '';

  if (!books || books.length === 0) {
    booksListEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📭</div>
        <p>No matching records found.</p>
      </div>`;
    booksCount.textContent = '';
    return;
  }

  booksCount.textContent = `${books.length} result${books.length !== 1 ? 's' : ''}`;

  books.forEach((book, i) => {
    const card = document.createElement('div');
    card.className = 'book-card';

    // Detect columns dynamically
    const title = pick(book, ['title', 'name', 'movie', 'book', 'product', 'item', 'subject']) || `Record ${i + 1}`;
    const author = pick(book, ['authors', 'author', 'director', 'creator', 'brand', 'artist']);
    const rating = pick(book, ['average_rating', 'rating', 'score', 'stars', 'grade']);
    const year = pick(book, ['published_year', 'year', 'release_year', 'date']);
    const pages = pick(book, ['num_pages', 'pages', 'length', 'duration']);
    const votes = pick(book, ['ratings_count', 'votes', 'reviews', 'count']);
    const cat = pick(book, ['categories', 'category', 'genre', 'type', 'department']);
    const desc = pick(book, ['description', 'summary', 'overview', 'about', 'content', 'plot']);

    const ratingStr = rating != null ? Number(rating).toFixed(2) : null;
    const stars = ratingStr ? starsHtml(parseFloat(ratingStr)) : '';

    // Build meta tags
    let metaHtml = '';
    if (ratingStr) metaHtml += `<span class="meta-tag rating">⭐ ${ratingStr} ${stars}</span>`;
    if (year != null) metaHtml += `<span class="meta-tag">📅 ${year}</span>`;
    if (pages != null) metaHtml += `<span class="meta-tag">📄 ${Number(pages)} pp</span>`;
    if (votes != null) metaHtml += `<span class="meta-tag">🗳️ ${Number(votes).toLocaleString()}</span>`;

    card.innerHTML = `
      <div class="book-rank">${i + 1}</div>
      <div class="book-title" title="${escapeAttr(title)}">${escapeHtml(title)}</div>
      ${author != null ? `<div class="book-authors">${escapeHtml(String(author))}</div>` : ''}
      ${metaHtml ? `<div class="book-meta">${metaHtml}</div>` : ''}
      ${cat != null ? `<div class="book-category">🏷️ ${escapeHtml(String(cat))}</div>` : ''}
      ${desc != null ? `<div class="book-desc">${escapeHtml(String(desc).slice(0, 220))}${String(desc).length > 220 ? '…' : ''}</div>` : ''}
    `;
    booksListEl.appendChild(card);
  });
}

function pick(obj, keys) {
  for (const k of keys) {
    const v = obj[k];
    if (v != null && v !== '') return v;
  }
  return null;
}

function starsHtml(rating) {
  const full = Math.floor(rating);
  const half = rating - full >= 0.5 ? 1 : 0;
  const empty = 5 - full - half;
  return '★'.repeat(full) + (half ? '½' : '') + '☆'.repeat(empty);
}


/* ══════════════════════════════════════════════════════════════
   UTILITIES
   ══════════════════════════════════════════════════════════════ */
function scrollToBottom() {
  requestAnimationFrame(() => { messagesEl.scrollTop = messagesEl.scrollHeight; });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
function escapeAttr(str) { return String(str).replace(/"/g, '&quot;'); }

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

let toastTimer;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 4000);
}

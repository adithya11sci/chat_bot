/* ── State ─────────────────────────────────────────────────────────────────── */
let isLoading = false;

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const messagesEl  = document.getElementById('messages');
const inputEl     = document.getElementById('user-input');
const sendBtn     = document.getElementById('send-btn');
const booksListEl = document.getElementById('books-list');
const booksCount  = document.getElementById('books-count');
const toastEl     = document.getElementById('toast');
const chipsRow    = document.getElementById('chips-row');

/* ── Auto-grow textarea ────────────────────────────────────────────────────── */
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

/* ── Keyboard: Enter sends, Shift+Enter newline ────────────────────────────── */
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isLoading) sendMessage(e);
  }
});

/* ── Fill query from chip ──────────────────────────────────────────────────── */
function fillQuery(text) {
  inputEl.value = text;
  inputEl.focus();
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

/* ── Send message ──────────────────────────────────────────────────────────── */
async function sendMessage(e) {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question || isLoading) return;

  // Hide chips after first message
  chipsRow.style.display = 'none';

  // Append user bubble
  appendMessage('user', question);
  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Show typing indicator
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

/* ── Append a chat bubble ──────────────────────────────────────────────────── */
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

/* ── Typing indicator ──────────────────────────────────────────────────────── */
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

/* ── Render book cards ─────────────────────────────────────────────────────── */
function renderBooks(books) {
  booksListEl.innerHTML = '';

  if (!books || books.length === 0) {
    booksListEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📭</div>
        <p>No specific books found for this query.</p>
      </div>`;
    booksCount.textContent = '';
    return;
  }

  booksCount.textContent = `${books.length} book${books.length !== 1 ? 's' : ''}`;

  books.forEach((book, i) => {
    const card = document.createElement('div');
    card.className = 'book-card';

    const rating   = book.average_rating != null ? Number(book.average_rating).toFixed(2) : 'N/A';
    const year     = book.published_year  != null ? book.published_year : '—';
    const pages    = book.num_pages       != null ? book.num_pages      : '—';
    const rCount   = book.ratings_count   != null ? Number(book.ratings_count).toLocaleString() : '—';
    const title    = book.title    || 'Untitled';
    const authors  = book.authors  || 'Unknown Author';
    const category = book.categories || '';
    const desc     = book.description || '';

    const stars = rating !== 'N/A' ? starsHtml(parseFloat(rating)) : '';

    card.innerHTML = `
      <div class="book-rank">${i + 1}</div>
      <div class="book-title" title="${escapeAttr(title)}">${escapeHtml(title)}</div>
      <div class="book-authors">${escapeHtml(authors)}</div>
      <div class="book-meta">
        <span class="meta-tag rating">⭐ ${rating} ${stars}</span>
        <span class="meta-tag">📅 ${year}</span>
        <span class="meta-tag">📄 ${pages} pp</span>
        <span class="meta-tag">🗳️ ${rCount} ratings</span>
      </div>
      ${category ? `<div class="book-category">🏷️ ${escapeHtml(category)}</div>` : ''}
      ${desc ? `<div class="book-desc">${escapeHtml(desc.slice(0, 220))}${desc.length > 220 ? '…' : ''}</div>` : ''}
    `;
    booksListEl.appendChild(card);
  });
}

/* ── Star rating helper ────────────────────────────────────────────────────── */
function starsHtml(rating) {
  const full  = Math.floor(rating);
  const half  = rating - full >= 0.5 ? 1 : 0;
  const empty = 5 - full - half;
  return '★'.repeat(full) + (half ? '½' : '') + '☆'.repeat(empty);
}

/* ── Utilities ─────────────────────────────────────────────────────────────── */
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

function setLoading(val) {
  isLoading = val;
  sendBtn.disabled = val;
  inputEl.disabled = val;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeAttr(str) {
  return String(str).replace(/"/g, '&quot;');
}

let toastTimer;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 4000);
}

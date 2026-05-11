/**
 * ML Customer Support Chatbot - Frontend
 */

const API = '/api';
let isLoading = false;
let currentStream = null;
let sessionId = localStorage.getItem('session_id');
let sessionStart = null;

if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem('session_id', sessionId);
}

// Elements
const messagesEl = document.getElementById('messages');
const form = document.getElementById('input-form');
const input = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.getElementById('sidebar');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toast-message');

// Init
document.addEventListener('DOMContentLoaded', async () => {
    input.focus();
    form.addEventListener('submit', handleSubmit);
    menuToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        sidebar.classList.toggle('open');
    });
    
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 968 && !sidebar.contains(e.target) && sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
        }
    });
    checkHealth();
    await loadSessionHistory();
});

// Health check
async function checkHealth() {
    try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();
        updateStatus(data);
    } catch (e) {
        updateStatus({ status: 'error' });
    }
}

function updateStatus(data) {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    const modelName = document.getElementById('model-name');
    const intentsCount = document.getElementById('intents-count');

    if (data.status === 'healthy') {
        dot.className = 'status-dot';
        text.textContent = 'Connected';
        modelName.textContent = data.model_name || '-';
        intentsCount.textContent = data.intents_count || '-';
    } else {
        dot.className = 'status-dot error';
        text.textContent = 'Disconnected';
    }
}

// Load existing session history on page load
async function loadSessionHistory() {
    try {
        const res = await fetch(`${API}/session`);
        if (!res.ok) return;
        const data = await res.json();
        sessionId = data.session_id;
        sessionStart = data.start_time;
        localStorage.setItem('session_id', sessionId);
        if (data.messages && data.messages.length > 0) {
            const welcome = document.querySelector('.welcome-message');
            if (welcome) welcome.remove();
            for (const msg of data.messages) {
                addMessage(msg.role, msg.content, {}, { noAnimation: true });
            }
        }
        updateSessionStats();
    } catch (e) {
        // server not ready yet, silently skip
    }
}

// Send message
async function sendMessage(message) {
    if (isLoading) return;
    isLoading = true;
    updateUI();

    if (currentStream) {
        cancelAnimationFrame(currentStream);
        currentStream = null;
    }

    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    addMessage('user', message);
    updateSessionStats();
    const typing = addTyping();

    try {
        const res = await fetch(`${API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message,
                session_id: sessionId 
            })
        });

        if (!res.ok) throw new Error('Failed to send message');

        const data = await res.json();
        typing.remove();
        sessionId = data.session_id;
        localStorage.setItem('session_id', sessionId);

        const bubbleEl = addMessage('assistant', data.response, {}, { stream: true });
        streamText(bubbleEl, data.response, {
            intent: data.intent,
            confidence: data.confidence,
            category: data.category,
        });
    } catch (e) {
        typing.remove();
        if (currentStream) {
            cancelAnimationFrame(currentStream);
            currentStream = null;
        }
        addMessage('assistant', 'Sorry, something went wrong. Please try again.');
        showToast('Failed to send message');
    } finally {
        isLoading = false;
        updateUI();
    }
}

// Quick message
function sendQuickMessage(msg) {
    input.value = msg;
    sendMessage(msg);
    input.value = '';
}

// Add message
function addMessage(role, text, meta = {}, options = {}) {
    const div = document.createElement('div');
    div.className = `message ${role}${options.noAnimation ? ' no-animation' : ''}`;

    const avatarHtml = role === 'user'
        ? '<img src="/static/assets/user.png" alt="User">'
        : '<img src="/static/assets/bot.png" alt="Bot">';

    let metaHtml = `<span class="message-time">${time()}</span>`;
    if (meta.intent) metaHtml += `<span class="message-tag">${meta.intent}</span>`;

    const rawAttr = text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const bubbleContent = options.stream ? '' : render(text);
    const metaSection = options.stream ? '' : `<div class="message-meta">${metaHtml}</div>`;

    div.innerHTML = `
        <div class="message-avatar">${avatarHtml}</div>
        <div class="message-content" data-raw="${options.stream ? '' : rawAttr}">
            <div class="message-bubble">${bubbleContent}</div>
            <button class="copy-btn" onclick="copyMessage(this)" aria-label="Copy message">
                <svg class="copy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                <svg class="check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><polyline points="20 6 9 17 4 12"/></svg>
            </button>
            ${metaSection}
        </div>
    `;

    messagesEl.appendChild(div);
    scroll();

    if (options.stream) {
        return div.querySelector('.message-bubble');
    }
}

// Typing indicator
function addTyping() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar">
            <img src="/static/assets/bot.png" alt="Bot">
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing"><span></span><span></span><span></span></div>
            </div>
        </div>
    `;
    messagesEl.appendChild(div);
    scroll();
    return div;
}

// Update prediction info
function updatePrediction(data) {
    document.getElementById('last-intent').textContent = data.intent || '-';
    document.getElementById('last-confidence').textContent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : '-';
    document.getElementById('last-category').textContent = data.category || '-';
}

// UI updates
function updateUI() {
    input.disabled = isLoading;
    sendBtn.disabled = isLoading;
}

function scroll() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function time() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function updateSessionStats() {
    const startEl = document.getElementById('session-start');
    const msgsEl = document.getElementById('session-msgs');
    if (!startEl || !msgsEl) return;
    if (sessionStart) {
        const d = new Date(sessionStart);
        startEl.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else {
        startEl.textContent = '-';
    }
    const count = document.querySelectorAll('.message:not(.no-animation), .message.no-animation').length;
    msgsEl.textContent = count;
}

function render(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    let html = div.innerHTML;

    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');

    html = html.replace(/((?:^|\n)[-*] .+(?:\n[-*] .+)*)/g, (match) => {
        const hasLeadingNewline = match.startsWith('\n');
        const lines = match.trim().split('\n').map(line => {
            const content = line.replace(/^[-*] /, '').trim();
            const rendered = content
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.+?)\*/g, '<em>$1</em>')
                .replace(/`(.+?)`/g, '<code>$1</code>');
            return '<li>' + rendered + '</li>';
        });
        return (hasLeadingNewline ? '\n' : '') + '<ul>' + lines.join('') + '</ul>';
    });

    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');
    html = html.replace(/\n/g, '<br>');
    return html;
}

function streamText(bubbleEl, fullText, meta) {
    let pos = 0;
    let lastTime = 0;
    const SPEED = 18;

    function tick(now) {
        if (!lastTime) lastTime = now;
        const elapsed = now - lastTime;
        const chars = Math.floor(elapsed / SPEED);

        if (pos < fullText.length) {
            pos = Math.min(pos + Math.max(1, chars), fullText.length);
            bubbleEl.innerHTML = render(fullText.slice(0, pos));
            const content = bubbleEl.closest('.message-content');
            if (content) {
                const partial = fullText.slice(0, pos);
                content.dataset.raw = partial.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            }
            scroll();
            lastTime = now;
            currentStream = requestAnimationFrame(tick);
        } else {
            currentStream = null;
            const content = bubbleEl.closest('.message-content');
            if (content) {
                content.dataset.raw = fullText.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                let metaHtml = `<span class="message-time">${time()}</span>`;
                if (meta.intent) metaHtml += `<span class="message-tag">${meta.intent}</span>`;
                const metaDiv = document.createElement('div');
                metaDiv.className = 'message-meta';
                metaDiv.innerHTML = metaHtml;
                content.appendChild(metaDiv);
            }
            updateSessionStats();
            updatePrediction(meta);
        }
    }

    currentStream = requestAnimationFrame(tick);
}

function renderWelcome() {
    const existing = document.querySelector('.welcome-message');
    if (existing) return;
    messagesEl.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <img src="/static/assets/bot.png" alt="" style="width:100%;height:100%;object-fit:cover;border-radius:50%;">
            </div>
            <h3>How can I help you today?</h3>
            <p>I can assist with orders, refunds, shipping, payments, and more.</p>
            <div class="quick-actions">
                <button class="quick-btn" onclick="sendQuickMessage('I want to cancel my order')" aria-label="Send: cancel order">Cancel Order</button>
                <button class="quick-btn" onclick="sendQuickMessage('Where is my refund?')" aria-label="Send: check refund">Check Refund</button>
                <button class="quick-btn" onclick="sendQuickMessage('Track my order')" aria-label="Send: track order">Track Order</button>
                <button class="quick-btn" onclick="sendQuickMessage('Payment methods')" aria-label="Send: payment info">Payment Info</button>
            </div>
        </div>
    `;
}

function copyMessage(btn) {
    const content = btn.closest('.message-content');
    const raw = content.dataset.raw;
    if (!raw) return;
    navigator.clipboard.writeText(raw).then(() => {
        btn.classList.add('copied');
        btn.querySelector('.copy-icon').style.display = 'none';
        btn.querySelector('.check-icon').style.display = '';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('.copy-icon').style.display = '';
            btn.querySelector('.check-icon').style.display = 'none';
        }, 2000);
    }).catch(() => {
        showToast('Failed to copy');
    });
}

function showToast(msg) {
    toastMessage.textContent = msg;
    toast.classList.add('show');
    setTimeout(hideToast, 4000);
}

function hideToast() {
    toast.classList.remove('show');
}

function autoResize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
}

// Submit handler
async function handleSubmit(e) {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg || isLoading) return;
    input.value = '';
    input.style.height = '';
    autoResize();
    await sendMessage(msg);
}

input.addEventListener('input', autoResize);

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
    }
});

// Reset session
async function resetSession() {
    if (currentStream) {
        cancelAnimationFrame(currentStream);
        currentStream = null;
    }
    try {
        const res = await fetch(`${API}/session/reset`, { method: 'POST' });
        if (!res.ok) throw new Error('Reset failed');
        const data = await res.json();
        sessionId = data.session_id;
        sessionStart = data.start_time;
        localStorage.setItem('session_id', sessionId);
        messagesEl.innerHTML = '';
        renderWelcome();
        updateSessionStats();
        showToast('New chat started');
    } catch (e) {
        showToast('Failed to reset session');
    }
}

document.getElementById('reset-btn').addEventListener('click', resetSession);

/**
 * ML Customer Support Chatbot - Frontend
 */

const API = '/api';
let isLoading = false;

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
document.addEventListener('DOMContentLoaded', () => {
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

// Send message
async function sendMessage(message) {
    if (isLoading) return;
    isLoading = true;
    updateUI();

    // Remove welcome message
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message
    addMessage('user', message);

    // Show typing
    const typing = addTyping();

    try {
        const res = await fetch(`${API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!res.ok) throw new Error('Failed to send message');

        const data = await res.json();
        typing.remove();
        addMessage('assistant', data.response, {
            intent: data.intent,
            confidence: data.confidence,
            category: data.category
        });
        updatePrediction(data);
    } catch (e) {
        typing.remove();
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
function addMessage(role, text, meta = {}) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const avatarHtml = role === 'user'
        ? '<img src="/static/assets/user.png" alt="User">'
        : '<img src="/static/assets/bot.png" alt="Bot">';

    let metaHtml = `<span class="message-time">${time()}</span>`;
    if (meta.intent) metaHtml += `<span class="message-tag">${meta.intent}</span>`;

    div.innerHTML = `
        <div class="message-avatar">${avatarHtml}</div>
        <div class="message-content">
            <div class="message-bubble">${escape(text)}</div>
            <div class="message-meta">${metaHtml}</div>
        </div>
    `;

    messagesEl.appendChild(div);
    scroll();
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

function escape(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

function showToast(msg) {
    toastMessage.textContent = msg;
    toast.classList.add('show');
    setTimeout(hideToast, 4000);
}

function hideToast() {
    toast.classList.remove('show');
}

// Submit handler
async function handleSubmit(e) {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg || isLoading) return;
    input.value = '';
    await sendMessage(msg);
}

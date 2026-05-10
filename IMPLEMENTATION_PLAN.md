# Implementation Plan: Conversation Context with Session Management

## Overview

Add conversation memory so the AI retains context across messages within a session. When a user says "my order ID is 12345" and then asks "can you refund that order", the AI understands "that order" refers to order 12345 without asking again.

**Scope:** Context window (last 10 messages), per-user session ID, in-memory storage. No database, no auth, no persistence across server restarts.

---

## Architecture

```
User opens page
    ↓
Frontend generates UUID → stores in localStorage as "session_id"
    ↓
Each message POST /api/chat { message, session_id }
    ↓
Backend looks up sessions[session_id] → creates if missing
    ↓
Appends user message to session
    ↓
Takes last 10 messages → builds LLM messages array
    ↓
Calls LLM with context array (not just current message)
    ↓
Appends assistant response to session
    ↓
Returns response + session persists in memory
```

---

## Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `backend/main.py` | Session store, context assembly, modified chat endpoint | ~30 |
| `frontend/script.js` | UUID generation, localStorage, send session_id | ~15 |

**Total: ~45 lines of new code.**

---

## Backend Changes (`backend/main.py`)

### 1. Global Session Store

Add near top of file (after imports):

```python
import uuid
from typing import Dict, List

# In-memory session store: { session_id: [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ] }
sessions: Dict[str, List[dict]] = {}
```

### 2. Update Request Model

Current `ChatRequest`:
```python
class ChatRequest(BaseModel):
    message: str
```

Updated:
```python
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
```

### 3. Update Chat Endpoint

Current flow:
```
receive message → classify intent → RAG retrieve → call LLM → return
```

New flow:
```
receive message + session_id
    ↓
resolve session (create new if missing)
    ↓
append user message to session
    ↓
classify intent → RAG retrieve
    ↓
build LLM messages: system prompt + last 10 messages from session
    ↓
call LLM
    ↓
append assistant response to session
    ↓
return response + session_id
```

Key code structure:

```python
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. Resolve session
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    
    # 2. Append user message
    sessions[session_id].append({"role": "user", "content": req.message})
    
    # 3. Classify intent + RAG (existing code, unchanged)
    intent, confidence = classify_intent(req.message)
    context = retrieve_context(req.message, intent)
    
    # 4. Build messages for LLM with context
    system_msg = {"role": "system", "content": f"You are a helpful customer support assistant. Intent: {intent}. Context: {context}"}
    history = sessions[session_id][-10:]  # Last 10 messages
    messages_for_llm = [system_msg] + history
    
    # 5. Call LLM
    response = await call_llm(messages_for_llm)
    
    # 6. Append assistant response
    sessions[session_id].append({"role": "assistant", "content": response})
    
    # 7. Return with session_id
    return ChatResponse(
        response=response,
        intent=intent,
        confidence=confidence,
        session_id=session_id
    )
```

### 4. Update Response Model

```python
class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    category: str
    session_id: str  # NEW
```

### 5. Optional: Session Cleanup Endpoint

```python
@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}
```

---

## Frontend Changes (`frontend/script.js`)

### 1. Session ID Management

Add at top of file:

```javascript
// Session management
function getSessionId() {
    let sid = localStorage.getItem('chat_session_id');
    if (!sid) {
        sid = crypto.randomUUID();
        localStorage.setItem('chat_session_id', sid);
    }
    return sid;
}

let currentSessionId = getSessionId();
```

### 2. Update sendMessage Function

Current:
```javascript
const res = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
});
```

Updated:
```javascript
const res = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
        message,
        session_id: currentSessionId
    })
});

// Update session_id if backend returned a new one
const data = await res.json();
if (data.session_id) {
    currentSessionId = data.session_id;
    localStorage.setItem('chat_session_id', currentSessionId);
}
```

### 3. Optional: Clear Session on "New Chat" Button

If you add a "New Chat" button later:
```javascript
function newChat() {
    currentSessionId = crypto.randomUUID();
    localStorage.setItem('chat_session_id', currentSessionId);
    messagesEl.innerHTML = ''; // Clear messages
    // Re-render welcome screen
}
```

---

## Context Window Behavior

| Scenario | What happens |
|----------|-------------|
| Message 1-10 | All messages sent to LLM |
| Message 11+ | Only last 10 sent, oldest dropped |
| Page refresh | Session preserved via localStorage + session_id |
| Server restart | Session lost, new UUID generated automatically |
| Multiple browser tabs | Each tab gets own session (separate UUIDs) |

---

## Example Conversation Flow

```
User: "My order ID is 12345"
→ sessions["abc-123"] = [
    {role: "user", content: "My order ID is 12345"}
  ]
→ LLM receives: [system, user: "My order ID is 12345"]
→ AI: "Got it! Order 12345. How can I help?"

User: "Can you refund that order?"
→ sessions["abc-123"] = [
    {role: "user", content: "My order ID is 12345"},
    {role: "assistant", content: "Got it! Order 12345. How can I help?"},
    {role: "user", content: "Can you refund that order?"}
  ]
→ LLM receives last 10 messages (3 total) → understands "that order" = 12345
→ AI: "I'll process a refund for order 12345. It will take 3-5 business days."
```

---

## Testing Checklist

- [ ] Send first message → new session_id created
- [ ] Send second message → same session_id, context carries over
- [ ] Refresh page → session_id preserved in localStorage
- [ ] Send message after refresh → same session continues
- [ ] Open new tab → separate session
- [ ] 11th message → oldest message dropped from context
- [ ] AI references previous context (order ID example)
- [ ] Server restart → new session starts cleanly

---

## Risk: In-Memory Growth

If server runs for days, sessions dict grows. For a demo (few hours), this is fine. If concerned, add:

```python
# Max sessions in memory
MAX_SESSIONS = 100

# In chat endpoint, before creating new session:
if len(sessions) >= MAX_SESSIONS:
    oldest = next(iter(sessions))
    del sessions[oldest]
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| New code | ~45 lines total |
| Files changed | 2 (`main.py`, `script.js`) |
| Database | None |
| Auth | None |
| Context window | Last 10 messages |
| Storage | Python dict (in-memory) |
| Session ID | UUID in localStorage |
| Server restart | Sessions cleared, new UUID auto-generated |

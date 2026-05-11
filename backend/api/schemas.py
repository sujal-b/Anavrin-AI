from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    start_time: datetime = Field(default_factory=datetime.now)
    messages: List[Dict[str, str]] = Field(default_factory=list)
    preferences: Dict[str, str] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = Field(None, description="Ignored; global session used")
    preferences: Optional[Dict[str, str]] = Field(None, description="Update session preferences")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v


class ChatResponse(BaseModel):
    user_message: str
    intent: str
    category: str
    confidence: float
    response: str
    session_id: str
    model_used: str
    timestamp: str
    preferences: Dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    intents_count: int
    load_time_ms: float
    timestamp: str

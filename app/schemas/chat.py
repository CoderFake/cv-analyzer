from dataclasses import field
from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.common import BaseSchema


class ChatMessageBase(BaseSchema):
    role: str  # "user", "assistant", "system"
    content: str
    message_metadata: Optional[Dict[str, Any]] = None


class ChatMessageCreate(BaseModel):
    role: str
    content: str
    message_metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class ChatMessage(ChatMessageBase):
    id: UUID
    chat_id: UUID
    created_at: Any


class ChatBase(BaseSchema):
    title: Optional[str] = None
    session_id: Optional[str] = None
    candidate_id: Optional[UUID] = None


class ChatCreate(BaseModel):
    title: Optional[str] = None
    session_id: Optional[str] = None
    candidate_id: Optional[UUID] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class ChatUpdate(BaseModel):
    title: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class Chat(ChatBase):
    id: UUID
    user_id: Optional[UUID] = None
    created_at: Any
    updated_at: Any
    messages: List[ChatMessage] = field(default_factory=list)


class ChatSummary(ChatBase):
    id: UUID
    user_id: Optional[UUID] = None
    created_at: Any
    updated_at: Any
    message_count: int = 0
    last_message: Optional[ChatMessage] = None


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[UUID] = None
    session_id: Optional[str] = None
    candidate_id: Optional[UUID] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class ChatResponse(BaseModel):
    chat_id: UUID
    message: ChatMessage
    candidate_id: Optional[UUID] = None

    model_config = ConfigDict(
        from_attributes=True
    )
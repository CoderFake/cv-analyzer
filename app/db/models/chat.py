import uuid
from sqlalchemy import Column, ForeignKey, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.models.base import BaseModel


class Chat(BaseModel):
    __tablename__ = "chats"

    title = Column(String, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    user = relationship("User", backref="chats")
    session_id = Column(String, nullable=True, index=True)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=True)
    candidate = relationship("Candidate", backref="chats")


class ChatMessage(BaseModel):
    __tablename__ = "chat_messages"

    chat_id = Column(UUID(as_uuid=True), ForeignKey("chats.id"), nullable=False)
    chat = relationship("Chat", backref="messages")
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=True)
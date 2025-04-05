from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models.chat import Chat, ChatMessage
from app.db.repositories.base import BaseRepository
from app.schemas.chat import ChatCreate, ChatUpdate, ChatMessageCreate


class ChatRepository(BaseRepository[Chat, ChatCreate, ChatUpdate]):
    def __init__(self, db: AsyncSession):
        super().__init__(Chat, db)

    async def create_chat(
            self,
            *,
            user_id: Optional[UUID] = None,
            session_id: Optional[str] = None,
            candidate_id: Optional[UUID] = None,
            title: Optional[str] = None
    ) -> Chat:
        chat = Chat(
            user_id=user_id,
            session_id=session_id,
            candidate_id=candidate_id,
            title=title or "New Chat"
        )
        self.db.add(chat)
        await self.db.commit()
        await self.db.refresh(chat)
        return chat

    async def add_message(
            self,
            *,
            chat_id: UUID,
            role: str,
            content: str,
            message_metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        if message_metadata == 'null':
            message_metadata = None

        if message_metadata is not None and not isinstance(message_metadata, dict):
            try:
                message_metadata = dict(message_metadata)
            except (TypeError, ValueError):
                message_metadata = None

        message = ChatMessage(
            chat_id=chat_id,
            role=role,
            content=content,
            message_metadata=message_metadata
        )
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def update_message_metadata(
            self,
            message_id: UUID,
            message_metadata: Dict[str, Any]
    ) -> Optional[ChatMessage]:
        """Cập nhật metadata của tin nhắn"""
        stmt = (
            update(ChatMessage)
            .where(ChatMessage.id == message_id)
            .values(message_metadata=message_metadata)
            .returning(ChatMessage)
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.scalars().first()

    async def get_message_with_metadata(
            self,
            message_id: UUID
    ) -> Optional[ChatMessage]:
        """Lấy thông tin tin nhắn kèm metadata"""
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.id == message_id)
        )
        return result.scalars().first()

    async def get_chat_with_messages(
            self,
            chat_id: UUID
    ) -> Optional[Chat]:
        result = await self.db.execute(
            select(Chat)
            .options(joinedload(Chat.messages))
            .where(Chat.id == chat_id)
        )
        return result.scalars().first()

    async def get_chats_by_user(
            self,
            user_id: UUID
    ) -> List[Tuple[Chat, Optional[ChatMessage]]]:
        chat_query = (
            select(Chat)
            .where(Chat.user_id == user_id)
            .order_by(Chat.updated_at.desc())
        )

        chat_result = await self.db.execute(chat_query)
        chats = chat_result.scalars().all()

        result = []

        for chat in chats:
            message_query = (
                select(ChatMessage)
                .where(ChatMessage.chat_id == chat.id)
                .order_by(ChatMessage.created_at.desc())
                .limit(1)
            )

            message_result = await self.db.execute(message_query)
            last_message = message_result.scalars().first()

            result.append((chat, last_message))

        return result

    async def get_chat_history(
            self,
            chat_id: UUID,
            limit: int = 50
    ) -> List[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_chats_by_session(
            self,
            session_id: str
    ) -> List[Tuple[Chat, Optional[ChatMessage]]]:
        chat_query = (
            select(Chat)
            .where(Chat.session_id == session_id)
            .order_by(Chat.updated_at.desc())
        )

        chat_result = await self.db.execute(chat_query)
        chats = chat_result.scalars().all()

        result = []
        for chat in chats:
            message_query = (
                select(ChatMessage)
                .where(ChatMessage.chat_id == chat.id)
                .order_by(ChatMessage.created_at.desc())
                .limit(1)
            )

            message_result = await self.db.execute(message_query)
            last_message = message_result.scalars().first()

            result.append((chat, last_message))

        return result

    async def delete_chat_with_messages(
            self,
            chat_id: UUID
    ) -> bool:

        await self.db.execute(
            delete(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
        )
        result = await self.delete(id=chat_id)
        return result is not None
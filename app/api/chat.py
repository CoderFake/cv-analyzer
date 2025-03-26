from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_user_or_session
from app.core.database import get_db
from app.db.repositories.candidate_repository import CandidateRepository
from app.db.repositories.chat_repository import ChatRepository
from app.schemas.chat import Chat, ChatMessage, ChatRequest, ChatResponse, ChatSummary
from app.schemas.common import ResponseBase
from app.services.llm_service import llm_service

router = APIRouter()


@router.post("/send", response_model=ResponseBase[ChatResponse])
async def send_message(
        chat_request: ChatRequest,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:

    chat_repo = ChatRepository(db)
    candidate_repo = CandidateRepository(db)

    chat = None
    candidate = None

    if chat_request.chat_id:
        chat = await chat_repo.get(id=chat_request.chat_id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )

        if (chat.user_id and chat.user_id != user_session["user_id"] and
                chat.session_id != user_session["session_id"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to this chat is not allowed"
            )

        if chat.candidate_id:
            candidate = await candidate_repo.get(id=chat.candidate_id)
    else:
        chat = await chat_repo.create_chat(
            user_id=user_session["user_id"],
            session_id=user_session["session_id"],
            candidate_id=chat_request.candidate_id,
            title=f"Chat {chat_request.message[:20]}..."
        )

        if chat_request.candidate_id:
            candidate = await candidate_repo.get(id=chat_request.candidate_id)

    user_message = await chat_repo.add_message(
        chat_id=chat.id,
        role="user",
        content=chat_request.message
    )

    chat_history = await chat_repo.get_chat_history(chat_id=chat.id)

    formatted_history = []
    for msg in chat_history:
        formatted_history.append({
            "role": msg.role,
            "content": msg.content
        })

    cv_content = None
    if candidate:
        cv_content = candidate.cv_content

    assistant_reply = await llm_service.chat_completion(
        question=chat_request.message,
        chat_history=formatted_history[:-1],
        cv_content=cv_content
    )

    assistant_message = await chat_repo.add_message(
        chat_id=chat.id,
        role="assistant",
        content=assistant_reply
    )

    assistant_message_data = ChatMessage.model_validate(assistant_message)

    return ResponseBase(
        success=True,
        data=ChatResponse(
            chat_id=chat.id,
            message=assistant_message_data,
            candidate_id=chat.candidate_id
        )
    )


@router.get("/list", response_model=ResponseBase[List[ChatSummary]])
async def list_chats(
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    chat_repo = ChatRepository(db)

    chat_data = []

    if user_session["user_id"]:
        chats = await chat_repo.get_chats_by_user(user_id=user_session["user_id"])
    else:
        chats = await chat_repo.get_chats_by_session(session_id=user_session["session_id"])

    for chat_tuple in chats:
        chat, last_message = chat_tuple
        chat_dict = Chat.model_validate(chat).model_dump()
        if last_message:
            chat_dict["last_message"] = ChatMessage.model_validate(last_message)
            chat_dict["message_count"] = 1
        else:
            chat_dict["message_count"] = 0

        chat_data.append(ChatSummary(**chat_dict))

    return ResponseBase(
        success=True,
        data=chat_data
    )


@router.get("/{chat_id}", response_model=ResponseBase[Chat])
async def get_chat(
        chat_id: UUID,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    chat_repo = ChatRepository(db)
    chat = await chat_repo.get_chat_with_messages(chat_id=chat_id)

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )

    if (chat.user_id and chat.user_id != user_session["user_id"] and
            chat.session_id != user_session["session_id"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this chat is not allowed"
        )

    chat_data = Chat.model_validate(chat)

    return ResponseBase(
        success=True,
        data=chat_data
    )


@router.delete("/{chat_id}", response_model=ResponseBase)
async def delete_chat(
        chat_id: UUID,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    chat_repo = ChatRepository(db)

    chat = await chat_repo.get(id=chat_id)

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )

    if (chat.user_id and chat.user_id != user_session["user_id"] and
            chat.session_id != user_session["session_id"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this chat is not allowed"
        )

    success = await chat_repo.delete_chat_with_messages(chat_id=chat_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat"
        )

    return ResponseBase(
        success=True,
        message="Chat deleted successfully"
    )
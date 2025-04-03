import os
import tempfile
import uuid
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_user_or_session
from app.core.database import get_db
from app.db.repositories.candidate_repository import CandidateRepository
from app.db.repositories.chat_repository import ChatRepository
from app.db.repositories.knowledge_repository import KnowledgeRepository
from app.schemas.chat import Chat, ChatMessage, ChatRequest, ChatResponse, ChatSummary
from app.schemas.common import ResponseBase
from app.services.integrated_llm_service import integrated_llm_service
from app.services.storage_service import storage_service
from app.utils.file_processor import FileProcessor
from app.services.context_classifier import context_classifier
from app.utils.cv_parser import CVParser

router = APIRouter()


@router.post("/message", response_model=ResponseBase[ChatResponse])
async def process_message(
        message: str = Form(...),
        chat_id: Optional[str] = Form(None),
        candidate_id: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:

    chat_repo = ChatRepository(db)
    candidate_repo = CandidateRepository(db)
    knowledge_repo = KnowledgeRepository(db)

    chat = None
    candidate = None
    knowledge_content = None
    file_content = None
    file_path_temp = None
    file_storage_path = None
    file_type = None
    file_name = None

    if file:
        try:
            folder = f"chat_uploads"
            if user_session["user_id"]:
                folder = f"{folder}/user_{user_session['user_id']}"
            else:
                folder = f"{folder}/session_{user_session['session_id']}"

            file_storage_path, file_name = await storage_service.upload_file(
                file=file.file,
                filename=file.filename,
                content_type=file.content_type,
                folder=folder
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.file.seek(0)
                content = await file.read()
                temp_file.write(content)
                file_path_temp = temp_file.name
                file_type = os.path.splitext(file.filename)[1]

            try:
                file_content = await CVParser.extract_text_from_file(file_path_temp)
            except Exception as e:
                print(f"Error extracting file content: {e}")
                file_content = f"Không thể đọc nội dung file: {str(e)}"
            finally:
                try:
                    if file_path_temp and os.path.exists(file_path_temp):
                        os.unlink(file_path_temp)
                except Exception as e:
                    print(f"Error removing temp file: {e}")
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            file_content = f"Không thể xử lý file: {str(e)}"
            file_storage_path = f"placeholder://uploads/{uuid.uuid4()}-{file.filename if file.filename else 'unknown.file'}"
            file_name = file.filename if file.filename else "unknown.file"

    if chat_id and chat_id != "null" and chat_id != "undefined":
        try:
            uuid_chat_id = UUID(chat_id)
            chat = await chat_repo.get(id=uuid_chat_id)
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
        except ValueError:
            chat = None

    uuid_candidate_id = None
    if candidate_id and candidate_id != "null" and candidate_id != "undefined":
        try:
            uuid_candidate_id = UUID(candidate_id)
            candidate = await candidate_repo.get(id=uuid_candidate_id)
        except ValueError:
            candidate = None

    if not chat:
        chat = await chat_repo.create_chat(
            user_id=user_session["user_id"],
            session_id=user_session["session_id"],
            candidate_id=uuid_candidate_id,
            title=message[:30] + "..." if len(message) > 30 else message
        )

    if chat.candidate_id and not candidate:
        candidate = await candidate_repo.get(id=chat.candidate_id)

    message_metadata = None
    if file and file_storage_path:
        message_metadata = {
            "file_path": file_storage_path,
            "file_name": file_name or file.filename
        }

    user_message_content = message
    if file and not message.strip():
        user_message_content = f"Tôi đã tải lên file: {file.filename}. Hãy phân tích file này giúp tôi."

    user_message = await chat_repo.add_message(
        chat_id=chat.id,
        role="user",
        content=user_message_content,
        message_metadata=message_metadata
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

    if context_classifier.should_use_knowledge_base(message):
        try:
            knowledge_docs = await knowledge_repo.search_knowledge(message)
            if knowledge_docs:
                knowledge_content = ""
                for i, doc in enumerate(knowledge_docs[:3], 1):
                    knowledge_content += f"Tài liệu {i} - {doc.title}:\n{doc.content}\n\n"
        except Exception as e:
            print(f"Error retrieving knowledge content: {e}")

    try:
        assistant_reply = ""

        if file and file_path_temp:
            assistant_reply = await integrated_llm_service.process_file_content(
                file_path=file_path_temp,
                file_content=file_content,
                file_type=file_type,
                query=message,
                cv_content=cv_content,
                knowledge_content=knowledge_content
            )
        else:
            assistant_reply = await integrated_llm_service.chat_with_knowledge(
                question=message,
                chat_history=formatted_history[:-1],
                cv_content=cv_content,
                knowledge_content=knowledge_content
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

    except Exception as e:
        error_message = f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu: {str(e)}"
        print(f"Error processing message: {e}")

        error_assistant_message = await chat_repo.add_message(
            chat_id=chat.id,
            role="assistant",
            content="Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
        )

        assistant_message_data = ChatMessage.model_validate(error_assistant_message)

        return ResponseBase(
            success=False,
            message=f"Error: {str(e)}",
            data=ChatResponse(
                chat_id=chat.id,
                message=assistant_message_data,
                candidate_id=chat.candidate_id
            )
        )
    finally:
        if file_path_temp and os.path.exists(file_path_temp):
            try:
                os.unlink(file_path_temp)
            except Exception as e:
                print(f"Error removing temp file: {e}")


@router.post("/send", response_model=ResponseBase[ChatResponse])
async def send_message(
        chat_request: ChatRequest,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:

    return await process_message(
        message=chat_request.message,
        chat_id=str(chat_request.chat_id) if chat_request.chat_id else None,
        candidate_id=str(chat_request.candidate_id) if chat_request.candidate_id else None,
        file=None,
        db=db,
        user_session=user_session
    )


@router.post("/send-file", response_model=ResponseBase[ChatResponse])
async def send_file_message(
        file: UploadFile = File(...),
        chat_id: Optional[str] = Form(None),
        candidate_id: Optional[str] = Form(None),
        message: str = Form(""),
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:

    return await process_message(
        message=message,
        chat_id=chat_id,
        candidate_id=candidate_id,
        file=file,
        db=db,
        user_session=user_session
    )


@router.get("/list", response_model=ResponseBase[List[ChatSummary]])
async def list_chats(
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    chat_repo = ChatRepository(db)

    chat_data = []

    try:
        if user_session["user_id"]:
            chats = await chat_repo.get_chats_by_user(user_id=user_session["user_id"])
        else:
            chats = await chat_repo.get_chats_by_session(session_id=user_session["session_id"])

        for chat, last_message in chats:
            chat_dict = {
                "id": chat.id,
                "title": chat.title,
                "user_id": chat.user_id,
                "session_id": chat.session_id,
                "candidate_id": chat.candidate_id,
                "created_at": chat.created_at,
                "updated_at": chat.updated_at,
                "message_count": 1 if last_message else 0
            }

            if last_message:
                chat_dict["last_message"] = ChatMessage.model_validate(last_message)

            chat_data.append(ChatSummary(**chat_dict))

        return ResponseBase(
            success=True,
            data=chat_data
        )
    except Exception as e:
        print(f"Error in list_chats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting chat list: {str(e)}"
        )

@router.get("/{chat_id}", response_model=ResponseBase[Chat])
async def get_chat(
        chat_id: UUID,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    """Lấy thông tin chi tiết một chat"""
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
    """Xóa một chat"""
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
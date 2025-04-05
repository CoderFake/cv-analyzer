import os
import tempfile
import uuid
from typing import Any, List, Optional, Dict
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
from app.services.storage_service import storage_service
from app.services.advanced_cv_analysis_service import AdvancedCVAnalysisService
from app.services.context_classifier import context_classifier

router = APIRouter()

# Khởi tạo dịch vụ phân tích CV
cv_analysis_service = AdvancedCVAnalysisService()


@router.post("/message", response_model=ResponseBase[ChatResponse])
async def process_message(
        message: str = Form(...),
        chat_id: Optional[str] = Form(None),
        candidate_id: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    """
    Xử lý tin nhắn từ người dùng, bao gồm cả tải lên và phân tích file CV
    """
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
    position_from_message = None

    # Thử trích xuất vị trí từ tin nhắn (nếu có)
    if "vị trí" in message.lower() or "position" in message.lower():
        import re
        position_match = re.search(r"vị trí[:\s]+([^\n.,?!]+)", message, re.IGNORECASE)
        if not position_match:
            position_match = re.search(r"position[:\s]+([^\n.,?!]+)", message, re.IGNORECASE)

        if position_match:
            position_from_message = position_match.group(1).strip()
            print(f"Extracted position from message: {position_from_message}")

    # Xử lý file tải lên (nếu có)
    if file:
        try:
            print(f"===== XỬ LÝ FILE TRONG API: {file.filename} =====")

            # Đảm bảo các thư mục tạm tồn tại
            os.makedirs('/tmp/chat_uploads', exist_ok=True)

            folder = f"chat_uploads"
            if user_session["user_id"]:
                folder = f"{folder}/user_{user_session['user_id']}"
            else:
                folder = f"{folder}/session_{user_session['session_id']}"

            # Lưu file trên cloud storage
            file.file.seek(0)
            file_storage_path, file_name = await storage_service.upload_file(
                file=file.file,
                filename=file.filename,
                content_type=file.content_type,
                folder=folder
            )
            print(f"Upload file thành công: {file_storage_path}")

            # Lưu file tạm vào hệ thống để xử lý
            file_extension = os.path.splitext(file.filename)[1].lower()
            print(f"File extension: {file_extension}")

            # Reset con trỏ file về đầu
            file.file.seek(0)

            # Tạo tên file tạm duy nhất
            temp_file_path = f"/tmp/chat_uploads/{uuid.uuid4()}{file_extension}"

            # Lưu file trực tiếp vào đường dẫn cụ thể
            content = await file.read()

            with open(temp_file_path, "wb") as f:
                f.write(content)

            file_path_temp = temp_file_path
            file_type = file_extension
            print(f"Đã lưu file tạm: {file_path_temp}, kích thước: {len(content)} bytes")

            # Kiểm tra file có tồn tại không
            if not os.path.exists(file_path_temp):
                raise Exception(f"File tạm không tồn tại: {file_path_temp}")
            else:
                print(f"Đã xác nhận file tạm tồn tại: {file_path_temp}")

        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            file_content = f"Không thể xử lý file: {str(e)}"
            file_storage_path = f"placeholder://uploads/{uuid.uuid4()}-{file.filename if file.filename else 'unknown.file'}"
            file_name = file.filename if file.filename else "unknown.file"

    # Tìm hoặc tạo chat
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
        user_message_content = f"Tôi đã tải lên file: {file.filename}. Hãy giúp tôi phân tích file cv này."

    # Lưu tin nhắn người dùng
    user_message = await chat_repo.add_message(
        chat_id=chat.id,
        role="user",
        content=user_message_content,
        message_metadata=message_metadata
    )

    # Lấy lịch sử chat
    chat_history = await chat_repo.get_chat_history(chat_id=chat.id)
    formatted_history = []
    for msg in chat_history:
        formatted_history.append({
            "role": msg.role,
            "content": msg.content
        })

    # Lấy nội dung CV từ candidate (nếu có)
    cv_content = None
    if candidate:
        cv_content = candidate.cv_content

    # Lấy thông tin knowledge base nếu cần
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

        # Kiểm tra xem đây có phải là phản hồi cho một yêu cầu vị trí trước đó không
        is_position_reply = False
        previous_message = formatted_history[-2] if len(formatted_history) >= 2 else None
        if previous_message and previous_message["role"] == "assistant" and (
                "vui lòng cho tôi biết vị trí" in previous_message["content"].lower() or
                "bạn đang ứng tuyển vị trí gì" in previous_message["content"].lower()):
            is_position_reply = True
            position = message.strip()
            print(f"Detected position reply with position: {position}")

            # Tìm file_path từ tin nhắn trước đó
            for msg in reversed(formatted_history[:-1]):  # Bỏ qua tin nhắn hiện tại
                if msg["role"] == "user" and "tôi đã tải lên file" in msg["content"].lower():
                    if len(formatted_history) >= 2:
                        # Tìm metadata của file từ tin nhắn gốc
                        prev_msg_id = None
                        for i, prev_msg in enumerate(chat_history):
                            if prev_msg.role == "user" and "tôi đã tải lên file" in prev_msg.content.lower():
                                prev_msg_id = prev_msg.id
                                break

                        if prev_msg_id:
                            prev_msg = await chat_repo.get_message_with_metadata(prev_msg_id)
                            if prev_msg and prev_msg.message_metadata:
                                file_info = prev_msg.message_metadata
                                if isinstance(file_info, str):
                                    import json
                                    try:
                                        file_info = json.loads(file_info)
                                    except:
                                        file_info = {}

                                temp_file_path = file_info.get("temp_file_path")
                                if temp_file_path and os.path.exists(temp_file_path):
                                    assistant_reply = await cv_analysis_service.continue_analysis_with_position(
                                        file_path=temp_file_path,
                                        file_type=os.path.splitext(temp_file_path)[1],
                                        position=position,
                                        chat_history=formatted_history
                                    )
                                    break

            # Nếu không tìm thấy file_path, trả lời thông thường
            if not assistant_reply:
                assistant_reply = f"Cảm ơn bạn đã cho biết vị trí {position}. Tuy nhiên, tôi không thể tìm thấy file CV đã tải lên trước đó. Vui lòng tải lại file CV để tôi có thể phân tích."

        # Xử lý file CV nếu có
        elif file_path_temp and os.path.exists(file_path_temp):
            print(f"===== PHÂN TÍCH FILE CV: {file_path_temp} =====")

            # Sử dụng dịch vụ advanced CV analysis
            position = position_from_message  # Từ tin nhắn hiện tại
            analysis_result, needs_position = await cv_analysis_service.process_cv_file(
                file_path=file_path_temp,
                file_type=file_type,
                position=position,
                chat_history=formatted_history[:-1]  # Bỏ qua tin nhắn hiện tại
            )

            # Cập nhật metadata của tin nhắn người dùng để lưu đường dẫn tệp tạm
            if needs_position:
                message_metadata = message_metadata or {}
                message_metadata["temp_file_path"] = file_path_temp
                await chat_repo.update_message_metadata(user_message.id, message_metadata)

            assistant_reply = analysis_result

        # Xử lý tin nhắn thông thường
        else:
            print("===== XỬ LÝ TIN NHẮN THÔNG THƯỜNG =====")

            # Kiểm tra xem có phải hỏi về phân tích CV không
            if "phân tích cv" in message.lower() or "đánh giá cv" in message.lower() or "review cv" in message.lower():
                assistant_reply = "Để tôi có thể phân tích CV của bạn, vui lòng tải lên tệp CV (định dạng PDF, DOCX, DOC hoặc hình ảnh). Nếu có thể, hãy cho tôi biết bạn đang ứng tuyển vị trí gì để tôi có thể cung cấp phân tích phù hợp nhất."
            else:
                # Gọi cv_analysis_service để trả lời câu hỏi thông thường
                assistant_reply = await cv_analysis_service.answer_general_question(
                    question=message,
                    chat_history=formatted_history[:-1],
                    cv_content=cv_content,
                    knowledge_content=knowledge_content
                )

        # Lưu câu trả lời của trợ lý
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
        # Không xóa file tạm nếu cần hỏi lại vị trí
        # File tạm sẽ được xóa sau khi xử lý xong
        pass


@router.post("/send", response_model=ResponseBase[ChatResponse])
async def send_message(
        chat_request: ChatRequest,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    """Gửi tin nhắn văn bản không kèm file"""
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
    """Gửi tin nhắn kèm file"""
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
    """Lấy danh sách các cuộc trò chuyện của người dùng"""
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
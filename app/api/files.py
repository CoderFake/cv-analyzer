from typing import Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_hr_user, get_user_or_session
from app.core.database import get_db
from app.db.repositories.candidate_repository import CandidateRepository
from app.schemas.common import ResponseBase
from app.schemas.user import User
from app.services.storage_service import storage_service
from app.utils.file_processor import FileProcessor

router = APIRouter()


@router.post("/upload", response_model=ResponseBase[dict])
async def upload_file(
        file: UploadFile = File(...),
        folder: str = Form("uploads"),
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    try:

        if user_session["user_id"]:
            folder = f"{folder}/user_{user_session['user_id']}"
        else:
            folder = f"{folder}/session_{user_session['session_id']}"

        file_path, file_name = await storage_service.upload_file(
            file=file.file,
            filename=file.filename,
            content_type=file.content_type,
            folder=folder
        )

        return ResponseBase(
            success=True,
            message="File uploaded successfully",
            data={
                "file_path": file_path,
                "file_name": file_name,
                "content_type": file.content_type
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@router.post("/upload-knowledge", response_model=ResponseBase[dict])
async def upload_knowledge_file(
        file: UploadFile = File(...),
        description: str = Form(None),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_hr_user)
) -> Any:
    try:
        folder = "knowledge"

        file_path, file_name = await storage_service.upload_file(
            file=file.file,
            filename=file.filename,
            content_type=file.content_type,
            folder=folder
        )

        # TODO: Lưu thông tin file vào database knowledge base nếu cần

        return ResponseBase(
            success=True,
            message="Knowledge file uploaded successfully",
            data={
                "file_path": file_path,
                "file_name": file_name,
                "description": description
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading knowledge file: {str(e)}"
        )


@router.get("/download/{candidate_id}", response_model=ResponseBase[dict])
async def get_download_url(
        candidate_id: UUID,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:

    candidate_repo = CandidateRepository(db)
    candidate = await candidate_repo.get(id=candidate_id)

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found"
        )

    if (candidate.user_id and candidate.user_id != user_session["user_id"] and
            candidate.session_id != user_session["session_id"] and
            user_session["role"] not in ["admin", "hr"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this candidate is not allowed"
        )

    file_path = candidate.cv_file_path

    parts = file_path.split("/")
    file_key = "/".join(parts[3:])

    download_url = await storage_service.get_presigned_url(file_key=file_key)

    if not download_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL"
        )

    return ResponseBase(
        success=True,
        data={
            "download_url": download_url,
            "filename": candidate.cv_file_name
        }
    )
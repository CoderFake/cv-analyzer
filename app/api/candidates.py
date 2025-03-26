from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_active_user, get_current_hr_user, get_user_or_session
from app.core.database import get_db
from app.db.models.candidate import GradeLevel
from app.db.repositories.candidate_repository import CandidateRepository
from app.schemas.candidate import Candidate, CVEvaluation
from app.schemas.common import ResponseBase
from app.schemas.user import User
from app.services.cv_analyzer import CVAnalyzerService
from app.utils.file_processor import FileProcessor

router = APIRouter()


@router.post("/upload", response_model=ResponseBase[Candidate])
async def upload_cv(
        file: UploadFile = File(...),
        position: Optional[str] = Form(None),
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    file_extension = FileProcessor.get_file_extension(file.filename)
    if file_extension not in FileProcessor.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File format not supported. Allowed formats: {', '.join(FileProcessor.ALLOWED_EXTENSIONS)}"
        )

    try:
        cv_data = await FileProcessor.process_cv_file(
            file=file,
            user_id=str(user_session["user_id"]) if user_session["user_id"] else None,
            session_id=user_session["session_id"]
        )

        candidate_repo = CandidateRepository(db)
        personal_info = cv_data.get("personal_info", {})

        candidate = await candidate_repo.create_with_cv(
            cv_file_path=cv_data["file_info"]["file_path"],
            cv_file_name=cv_data["file_info"]["filename"],
            cv_file_extension=cv_data["file_info"]["extension"],
            fullname=personal_info.get("name"),
            email=personal_info.get("email"),
            phone=personal_info.get("phone"),
            position_applied=position,
            user_id=user_session["user_id"],
            session_id=user_session["session_id"]
        )

        await candidate_repo.update_cv_data(
            candidate_id=candidate.id,
            cv_content=cv_data["cv_text"],
            cv_data=cv_data["cv_data"]
        )
        candidate_data = Candidate.model_validate(candidate)

        return ResponseBase(
            success=True,
            message="CV uploaded successfully",
            data=candidate_data
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing CV: {str(e)}"
        )


@router.get("/", response_model=ResponseBase[List[Candidate]])
async def list_candidates(
        db: AsyncSession = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_active_user),
        user_session: dict = Depends(get_user_or_session)
) -> Any:
    candidate_repo = CandidateRepository(db)

    if user_session["user_id"]:
        candidates = await candidate_repo.get_by_user(user_id=user_session["user_id"])
    else:
        candidates = await candidate_repo.get_by_session(session_id=user_session["session_id"])

    candidate_data = [Candidate.model_validate(c) for c in candidates]

    return ResponseBase(
        success=True,
        data=candidate_data
    )


@router.get("/all", response_model=ResponseBase[List[Candidate]])
async def list_all_candidates(
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_hr_user)
) -> Any:

    candidate_repo = CandidateRepository(db)
    candidates = await candidate_repo.get_multi(limit=1000)

    candidate_data = [Candidate.model_validate(c) for c in candidates]

    return ResponseBase(
        success=True,
        data=candidate_data
    )


@router.get("/{candidate_id}", response_model=ResponseBase[Candidate])
async def get_candidate(
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

    candidate_data = Candidate.model_validate(candidate)

    return ResponseBase(
        success=True,
        data=candidate_data
    )


@router.post("/{candidate_id}/evaluate", response_model=ResponseBase[CVEvaluation])
async def evaluate_candidate(
        candidate_id: UUID,
        db: AsyncSession = Depends(get_db),
        user_session: dict = Depends(get_user_or_session),
        cv_analyzer_service: CVAnalyzerService = Depends(lambda: CVAnalyzerService(db))
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

    evaluation = await cv_analyzer_service.evaluate_cv(candidate)

    await candidate_repo.update_evaluation(
        candidate_id=candidate.id,
        grade=evaluation.grade,
        evaluation=evaluation.evaluation
    )

    return ResponseBase(
        success=True,
        message="CV evaluated successfully",
        data=evaluation
    )


@router.put("/{candidate_id}", response_model=ResponseBase[Candidate])
async def update_candidate(
        candidate_id: UUID,
        grade: Optional[GradeLevel] = None,
        evaluation: Optional[str] = None,
        position_applied: Optional[str] = None,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_hr_user)
) -> Any:
    candidate_repo = CandidateRepository(db)
    candidate = await candidate_repo.get(id=candidate_id)

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found"
        )

    update_data = {}
    if grade is not None:
        update_data["grade"] = grade
    if evaluation is not None:
        update_data["evaluation"] = evaluation
    if position_applied is not None:
        update_data["position_applied"] = position_applied

    if update_data:
        candidate = await candidate_repo.update(id=candidate_id, obj_in=update_data)

    candidate_data = Candidate.model_validate(candidate)

    return ResponseBase(
        success=True,
        message="Candidate updated successfully",
        data=candidate_data
    )


@router.delete("/{candidate_id}", response_model=ResponseBase)
async def delete_candidate(
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

    await candidate_repo.delete(id=candidate_id)

    return ResponseBase(
        success=True,
        message="Candidate deleted successfully"
    )
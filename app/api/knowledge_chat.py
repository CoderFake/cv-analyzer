from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_active_user
from app.core.database import get_db
from app.schemas.common import ResponseBase
from app.schemas.user import User
from app.services.llm_knowledge_service import LLMKnowledgeService

router = APIRouter()


@router.post("/query", response_model=ResponseBase[dict])
async def knowledge_query(
        question: str,
        category: Optional[str] = None,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:
    if not question or len(question) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 3 characters"
        )

    llm_knowledge_service = LLMKnowledgeService(db)
    result = await llm_knowledge_service.answer_with_knowledge(
        question=question,
        category=category
    )

    return ResponseBase(
        success=True,
        data=result
    )


@router.get("/summary", response_model=ResponseBase[str])
async def knowledge_summary(
        category: Optional[str] = None,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:
    llm_knowledge_service = LLMKnowledgeService(db)
    summary = await llm_knowledge_service.generate_knowledge_summary(category=category)

    return ResponseBase(
        success=True,
        data=summary
    )
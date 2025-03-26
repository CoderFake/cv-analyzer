from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.candidate import Candidate, GradeLevel
from app.db.repositories.base import BaseRepository
from app.schemas.candidate import CandidateCreate, CandidateUpdate


class CandidateRepository(BaseRepository[Candidate, CandidateCreate, CandidateUpdate]):
    def __init__(self, db: AsyncSession):
        super().__init__(Candidate, db)

    async def create_with_cv(
            self,
            *,
            cv_file_path: str,
            cv_file_name: str,
            cv_file_extension: str,
            user_id: Optional[UUID] = None,
            session_id: Optional[str] = None,
            **candidate_data
    ) -> Candidate:
        db_obj = Candidate(
            cv_file_path=cv_file_path,
            cv_file_name=cv_file_name,
            cv_file_extension=cv_file_extension,
            user_id=user_id,
            session_id=session_id,
            **candidate_data
        )
        self.db.add(db_obj)
        await self.db.commit()
        await self.db.refresh(db_obj)
        return db_obj

    async def update_cv_data(
            self,
            *,
            candidate_id: UUID,
            cv_content: str,
            cv_data: Dict[str, Any]
    ) -> Optional[Candidate]:
        return await self.update(
            id=candidate_id,
            obj_in={
                "cv_content": cv_content,
                "cv_data": cv_data
            }
        )

    async def update_evaluation(
            self,
            *,
            candidate_id: UUID,
            grade: GradeLevel,
            evaluation: str
    ) -> Optional[Candidate]:
        return await self.update(
            id=candidate_id,
            obj_in={
                "grade": grade,
                "evaluation": evaluation
            }
        )

    async def get_by_session(self, session_id: str) -> List[Candidate]:
        result = await self.db.execute(
            select(Candidate)
            .where(Candidate.session_id == session_id)
            .order_by(Candidate.created_at.desc())
        )
        return result.scalars().all()

    async def get_by_user(self, user_id: UUID) -> List[Candidate]:
        result = await self.db.execute(
            select(Candidate)
            .where(Candidate.user_id == user_id)
            .order_by(Candidate.created_at.desc())
        )
        return result.scalars().all()

    async def get_stats_by_grade(self) -> Dict[str, int]:
        result = await self.db.execute(
            select(Candidate.grade, func.count(Candidate.id))
            .where(Candidate.grade.is_not(None))
            .group_by(Candidate.grade)
        )
        return {grade.value: count for grade, count in result.all()}
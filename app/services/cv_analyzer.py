from typing import Dict, Any, Optional
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.candidate import Candidate, GradeLevel
from app.schemas.candidate import CVEvaluation
from app.services.llm_service import llm_service
from app.services.web_search_service import web_search_service


class CVAnalyzerService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def evaluate_cv(self, candidate: Candidate) -> CVEvaluation:
        cv_content = candidate.cv_content

        position = candidate.position_applied or "Không xác định"

        search_query = f"Yêu cầu tuyển dụng vị trí {position} và kỹ năng cần thiết"
        search_results = await web_search_service.process_search_results(search_query)

        search_text = ""
        for idx, result in enumerate(search_results.get("search_results", []), 1):
            search_text += f"{idx}. {result['title']}: {result['content']}\n\n"

        evaluation_result = await llm_service.evaluate_cv(
            cv_content=cv_content,
            position=position,
            use_search=True
        )

        grade = self._parse_grade(evaluation_result)
        cv_evaluation = CVEvaluation(
            grade=grade,
            evaluation=evaluation_result.get("evaluation", ""),
            summary=evaluation_result.get("summary", ""),
            strengths=evaluation_result.get("strengths", []),
            weaknesses=evaluation_result.get("weaknesses", []),
            recommendations=evaluation_result.get("recommendations", [])
        )

        return cv_evaluation

    def _parse_grade(self, evaluation_result: Dict[str, Any]) -> GradeLevel:
        grade_str = evaluation_result.get("grade", "C").upper()
        grade_map = {
            "A": GradeLevel.A,
            "B": GradeLevel.B,
            "C": GradeLevel.C,
            "D": GradeLevel.D,
            "E": GradeLevel.E
        }

        return grade_map.get(grade_str, GradeLevel.C)
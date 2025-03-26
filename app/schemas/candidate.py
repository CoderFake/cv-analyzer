from dataclasses import field
from typing import Dict, Optional, Any, List
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, ConfigDict

from app.db.models.candidate import GradeLevel
from app.schemas.common import BaseSchema


class CVData(BaseModel):
    personal_info: Dict[str, Any]
    education: List[Dict[str, Any]]
    work_experience: List[Dict[str, Any]]
    skills: List[str]
    languages: List[Dict[str, Any]]
    certifications: List[Dict[str, Any]]
    projects: List[Dict[str, Any]]
    achievements: List[Dict[str, Any]]
    additional_info: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class CandidateBase(BaseSchema):
    fullname: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    position_applied: Optional[str] = None

    cv_file_name: str
    cv_file_extension: str


class CandidateCreate(BaseModel):
    fullname: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    position_applied: Optional[str] = None
    session_id: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class CandidateUpdate(BaseModel):
    fullname: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    position_applied: Optional[str] = None
    grade: Optional[GradeLevel] = None
    evaluation: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class Candidate(CandidateBase):
    id: UUID
    cv_file_path: str
    grade: Optional[GradeLevel] = None
    evaluation: Optional[str] = None
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    created_at: Any
    updated_at: Any


class CVEvaluation(BaseModel):
    grade: GradeLevel
    evaluation: str
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    market_trends: List[str] = field(default_factory=list)
    recommendations: List[str]

    model_config = ConfigDict(
        from_attributes=True
    )
from enum import Enum
import uuid
from sqlalchemy import Column, ForeignKey, String, JSON, Enum as SQLEnum, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.models.base import BaseModel


class GradeLevel(str, Enum):
    A = "A"  # Xuất sắc
    B = "B"  # Tốt
    C = "C"  # Trung bình
    D = "D"  # Dưới trung bình
    E = "E"  # Không phù hợp


class Candidate(BaseModel):

    __tablename__ = "candidates"

    fullname = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    position_applied = Column(String, nullable=True)

    cv_file_path = Column(String, nullable=False)
    cv_file_name = Column(String, nullable=False)
    cv_file_extension = Column(String, nullable=False)

    cv_content = Column(Text, nullable=True)
    cv_data = Column(JSON, nullable=True)

    grade = Column(SQLEnum(GradeLevel), nullable=True)
    evaluation = Column(Text, nullable=True)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    user = relationship("User", backref="candidates")

    session_id = Column(String, nullable=True)
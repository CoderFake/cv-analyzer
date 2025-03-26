from enum import Enum
from sqlalchemy import Boolean, Column, String, Enum as SQLEnum

from app.db.models.base import BaseModel


class UserRole(str, Enum):
    ADMIN = "admin"
    HR = "hr"
    USER = "user"


class User(BaseModel):
    __tablename__ = "users"

    email = Column(String, unique=True, index=True, nullable=False)
    fullname = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
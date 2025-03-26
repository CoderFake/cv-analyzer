from typing import Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, ConfigDict

from app.db.models.user import UserRole
from app.schemas.common import BaseSchema


class UserBase(BaseSchema):
    email: EmailStr
    fullname: str
    role: UserRole = UserRole.USER
    is_active: bool = True


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    fullname: str
    role: Optional[UserRole] = UserRole.USER

    model_config = ConfigDict(
        from_attributes=True
    )


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    fullname: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class User(UserBase):
    id: UUID


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    model_config = ConfigDict(
        from_attributes=True
    )


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User

    model_config = ConfigDict(
        from_attributes=True
    )


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    role: Optional[UserRole] = None
    exp: Optional[int] = None

    model_config = ConfigDict(
        from_attributes=True
    )
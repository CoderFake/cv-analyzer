from typing import Generator, Optional
from uuid import UUID
import re
import uuid

from fastapi import Depends, HTTPException, status, Cookie, Request
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import validate_token
from app.db.models.user import UserRole
from app.db.repositories.user_repository import UserRepository
from app.schemas.user import TokenPayload, User

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=False
)

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def get_current_user(
        db: AsyncSession = Depends(get_db),
        token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[User]:
    if not token:
        return None

    try:
        payload = validate_token(token)
        if payload is None:
            return None

        token_data = TokenPayload.model_validate(payload)

        if token_data.exp is None:
            return None

    except (jwt.JWTError, ValidationError):
        return None

    user_repo = UserRepository(db)
    user = await user_repo.get(UUID(token_data.sub))

    if not user:
        return None

    if not user.is_active:
        return None

    return User.model_validate(user)


async def get_current_active_user(
        current_user: Optional[User] = Depends(get_current_user),
) -> User:
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


async def get_current_admin_user(
        current_user: User = Depends(get_current_active_user),
) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user


async def get_current_hr_user(
        current_user: User = Depends(get_current_active_user),
) -> User:
    if current_user.role not in [UserRole.HR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user


async def get_session_id(
        request: Request,
        session_id: Optional[str] = Cookie(None)
) -> str:
    if session_id and SESSION_ID_PATTERN.match(session_id):
        return session_id

    return str(uuid.uuid4())


async def get_user_or_session(
        db: AsyncSession = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
        session_id: str = Depends(get_session_id)
) -> dict:
    if current_user:
        return {
            "user_id": current_user.id,
            "session_id": None,
            "is_authenticated": True,
            "role": current_user.role
        }
    else:
        return {
            "user_id": None,
            "session_id": session_id,
            "is_authenticated": False,
            "role": None
        }
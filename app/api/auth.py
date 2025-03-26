from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_current_active_user
from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_access_token
from app.db.repositories.user_repository import UserRepository
from app.schemas.user import User, UserCreate, Token, LoginRequest
from app.schemas.common import ResponseBase

router = APIRouter()


@router.post("/login", response_model=ResponseBase[Token])
async def login(
        response: Response,
        form_data: LoginRequest,
        db: AsyncSession = Depends(get_db)
) -> Any:
    user_repo = UserRepository(db)
    user = await user_repo.authenticate(email=form_data.email, password=form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "role": user.role
    }

    access_token = create_access_token(
        subject=str(user.id),
        expires_delta=access_token_expires,
        payload=payload
    )

    user_data = User.model_validate(user)

    return ResponseBase(
        success=True,
        message="Login successful",
        data=Token(
            access_token=access_token,
            token_type="bearer",
            user=user_data
        )
    )


@router.post("/register", response_model=ResponseBase[User])
async def register(
        user_in: UserCreate,
        db: AsyncSession = Depends(get_db),
) -> Any:
    user_repo = UserRepository(db)

    existing_user = await user_repo.get_by_email(email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = await user_repo.create(obj_in=user_in)
    user_data = User.model_validate(user)

    return ResponseBase(
        success=True,
        message="User registered successfully",
        data=user_data
    )


@router.get("/me", response_model=ResponseBase[User])
async def read_users_me(
        current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user
    """
    return ResponseBase(
        success=True,
        data=current_user
    )


@router.post("/logout", response_model=ResponseBase)
async def logout(
    response: Response,
    current_user: User = Depends(get_current_user),
) -> Any:

    return ResponseBase(
        success=True,
        message="Logout successful"
    )
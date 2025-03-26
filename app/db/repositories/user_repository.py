from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User
from app.db.repositories.base import BaseRepository
from app.schemas.user import UserCreate, UserUpdate
from app.core.security import get_password_hash, verify_password


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    def __init__(self, db: AsyncSession):
        super().__init__(User, db)

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalars().first()

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        user = await self.get_by_email(email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def create(self, *, obj_in: UserCreate) -> User:
        user_data = obj_in.model_dump(exclude={"password"})

        db_obj = User(
            **user_data,
            hashed_password=get_password_hash(obj_in.password)
        )
        self.db.add(db_obj)
        await self.db.commit()
        await self.db.refresh(db_obj)
        return db_obj

    async def update_password(self, *, user_id: UUID, new_password: str) -> Optional[User]:
        user = await self.get(id=user_id)
        if not user:
            return None

        hashed_password = get_password_hash(new_password)
        return await self.update(id=user_id, obj_in={"hashed_password": hashed_password})
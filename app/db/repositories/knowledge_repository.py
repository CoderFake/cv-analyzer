from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.knowledge import KnowledgeDocument
from app.db.repositories.base import BaseRepository
from app.schemas.knowledge import KnowledgeDocumentCreate, KnowledgeDocumentUpdate


class KnowledgeRepository(BaseRepository[KnowledgeDocument, KnowledgeDocumentCreate, KnowledgeDocumentUpdate]):

    def __init__(self, db: AsyncSession):
        super().__init__(KnowledgeDocument, db)

    async def create_knowledge_document(
            self,
            *,
            title: str,
            file_path: str,
            file_name: str,
            file_extension: str,
            uploaded_by: UUID,
            description: Optional[str] = None,
            content: Optional[str] = None,
            category: Optional[str] = None
    ) -> KnowledgeDocument:
        knowledge_doc = KnowledgeDocument(
            title=title,
            description=description,
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            content=content,
            uploaded_by=uploaded_by,
            category=category,
            is_active=True
        )
        self.db.add(knowledge_doc)
        await self.db.commit()
        await self.db.refresh(knowledge_doc)
        return knowledge_doc

    async def get_by_category(self, category: str) -> List[KnowledgeDocument]:
        result = await self.db.execute(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.category == category, KnowledgeDocument.is_active == True)
            .order_by(KnowledgeDocument.created_at.desc())
        )
        return result.scalars().all()

    async def get_by_uploader(self, uploaded_by: UUID) -> List[KnowledgeDocument]:
        result = await self.db.execute(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.uploaded_by == uploaded_by)
            .order_by(KnowledgeDocument.created_at.desc())
        )
        return result.scalars().all()

    async def search_knowledge(self, query: str) -> List[KnowledgeDocument]:
        search_term = f"%{query}%"
        result = await self.db.execute(
            select(KnowledgeDocument)
            .where(
                (KnowledgeDocument.title.ilike(search_term)) |
                (KnowledgeDocument.description.ilike(search_term)) |
                (KnowledgeDocument.content.ilike(search_term))
            )
            .where(KnowledgeDocument.is_active == True)
            .order_by(KnowledgeDocument.created_at.desc())
        )
        return result.scalars().all()

    async def update_content(
            self,
            *,
            knowledge_id: UUID,
            content: str
    ) -> Optional[KnowledgeDocument]:
        return await self.update(
            id=knowledge_id,
            obj_in={"content": content}
        )

    async def deactivate(self, *, knowledge_id: UUID) -> Optional[KnowledgeDocument]:
        return await self.update(
            id=knowledge_id,
            obj_in={"is_active": False}
        )
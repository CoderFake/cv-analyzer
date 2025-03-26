from typing import Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.common import BaseSchema


class KnowledgeDocumentBase(BaseSchema):
    title: str
    description: Optional[str] = None
    file_name: str
    file_extension: str
    category: Optional[str] = None


class KnowledgeDocumentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    file_path: str
    file_name: str
    file_extension: str
    content: Optional[str] = None
    category: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class KnowledgeDocumentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class KnowledgeDocument(KnowledgeDocumentBase):
    id: UUID
    file_path: str
    uploaded_by: UUID
    is_active: bool
    created_at: Any
    updated_at: Any
    content_preview: Optional[str] = None
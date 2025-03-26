from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar('T')

class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: lambda id: str(id),
        }
    )

class ResponseBase(BaseSchema, Generic[T]):
    """Base response schema"""
    success: bool = True
    message: Optional[str] = None
    data: Optional[T] = None

class PaginatedResponse(ResponseBase[List[T]], Generic[T]):
    """Paginated response schema"""
    total: int
    page: int
    size: int
    pages: int

class ErrorResponse(BaseSchema):
    """Error response schema"""
    success: bool = False
    error: str
    detail: Optional[Any] = None
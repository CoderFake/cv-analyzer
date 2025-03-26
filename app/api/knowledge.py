from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_hr_user, get_current_active_user
from app.core.database import get_db
from app.db.repositories.knowledge_repository import KnowledgeRepository
from app.schemas.knowledge import KnowledgeDocument, KnowledgeDocumentUpdate
from app.schemas.common import ResponseBase
from app.schemas.user import User
from app.services.storage_service import storage_service
from app.utils.file_processor import FileProcessor

router = APIRouter()


@router.post("/upload", response_model=ResponseBase[KnowledgeDocument])
async def upload_knowledge(
        file: UploadFile = File(...),
        title: str = Form(...),
        description: Optional[str] = Form(None),
        category: Optional[str] = Form(None),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_hr_user)
) -> Any:

    try:
        file_extension = FileProcessor.get_file_extension(file.filename)
        if file_extension not in ['.pdf', '.docx', '.doc', '.txt']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File format not supported. Allowed formats: .pdf, .docx, .doc, .txt"
            )

        folder = f"knowledge/{category or 'general'}"
        file_path, file_name = await storage_service.upload_file(
            file=file.file,
            filename=file.filename,
            content_type=file.content_type,
            folder=folder
        )

        file.file.seek(0)
        with open(f"/tmp/{file.filename}", "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        try:
            from app.utils.cv_parser import CVParser
            file_content = await CVParser.extract_text_from_file(f"/tmp/{file.filename}")
        except Exception as e:
            file_content = f"Error extracting content: {str(e)}"
            print(f"Error extracting content from file: {str(e)}")

        knowledge_repo = KnowledgeRepository(db)
        knowledge_doc = await knowledge_repo.create_knowledge_document(
            title=title,
            description=description,
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            content=file_content,
            category=category,
            uploaded_by=current_user.id
        )

        knowledge_doc_data = KnowledgeDocument.model_validate(knowledge_doc)
        knowledge_doc_data_dict = knowledge_doc_data.model_dump()
        knowledge_doc_data_dict["content_preview"] = file_content[:500] + "..." if len(
            file_content) > 500 else file_content
        knowledge_doc_data_dict.pop("content", None)

        return ResponseBase(
            success=True,
            message="Knowledge document uploaded successfully",
            data=knowledge_doc_data_dict
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading knowledge document: {str(e)}"
        )


@router.get("/", response_model=ResponseBase[List[KnowledgeDocument]])
async def list_knowledge_documents(
        category: Optional[str] = Query(None),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:
    knowledge_repo = KnowledgeRepository(db)

    if category:
        knowledge_docs = await knowledge_repo.get_by_category(category)
    else:
        knowledge_docs = await knowledge_repo.get_multi(is_active=True)

    result = []
    for doc in knowledge_docs:
        doc_data = KnowledgeDocument.model_validate(doc)
        doc_dict = doc_data.model_dump()
        if doc.content:
            doc_dict["content_preview"] = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
        doc_dict.pop("content", None)
        result.append(doc_dict)

    return ResponseBase(
        success=True,
        data=result
    )


@router.get("/search", response_model=ResponseBase[List[KnowledgeDocument]])
async def search_knowledge(
        query: str,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:

    if not query or len(query) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query must be at least 3 characters"
        )

    knowledge_repo = KnowledgeRepository(db)
    knowledge_docs = await knowledge_repo.search_knowledge(query)

    result = []
    for doc in knowledge_docs:
        doc_data = KnowledgeDocument.model_validate(doc)
        doc_dict = doc_data.model_dump()
        if doc.content:
            doc_dict["content_preview"] = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
        doc_dict.pop("content", None)
        result.append(doc_dict)

    return ResponseBase(
        success=True,
        data=result
    )


@router.get("/{knowledge_id}", response_model=ResponseBase[KnowledgeDocument])
async def get_knowledge_document(
        knowledge_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:

    knowledge_repo = KnowledgeRepository(db)
    knowledge_doc = await knowledge_repo.get(id=knowledge_id)

    if not knowledge_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge document not found"
        )

    if not knowledge_doc.is_active and current_user.role not in ["admin", "hr"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Knowledge document is not active"
        )

    knowledge_data = KnowledgeDocument.model_validate(knowledge_doc)

    return ResponseBase(
        success=True,
        data=knowledge_data
    )


@router.put("/{knowledge_id}", response_model=ResponseBase[KnowledgeDocument])
async def update_knowledge_document(
        knowledge_id: UUID,
        update_data: KnowledgeDocumentUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_hr_user)
) -> Any:

    knowledge_repo = KnowledgeRepository(db)
    knowledge_doc = await knowledge_repo.get(id=knowledge_id)

    if not knowledge_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge document not found"
        )

    knowledge_doc = await knowledge_repo.update(id=knowledge_id, obj_in=update_data)

    knowledge_data = KnowledgeDocument.model_validate(knowledge_doc)

    return ResponseBase(
        success=True,
        message="Knowledge document updated successfully",
        data=knowledge_data
    )


@router.delete("/{knowledge_id}", response_model=ResponseBase)
async def delete_knowledge_document(
        knowledge_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_hr_user)
) -> Any:

    knowledge_repo = KnowledgeRepository(db)
    knowledge_doc = await knowledge_repo.get(id=knowledge_id)

    if not knowledge_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge document not found"
        )

    await knowledge_repo.deactivate(knowledge_id=knowledge_id)

    return ResponseBase(
        success=True,
        message="Knowledge document deleted successfully"
    )
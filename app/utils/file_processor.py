import os
import tempfile
import mimetypes
from typing import BinaryIO, Dict, Any, Tuple, Optional
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.utils.cv_parser import CVParser
from app.services.storage_service import storage_service


class FileProcessor:
    ALLOWED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt', '.jpg', '.jpeg', '.png')

    @staticmethod
    async def process_cv_file(
            file: UploadFile,
            user_id: Optional[str] = None,
            session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in FileProcessor.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File format not supported. Allowed formats: {', '.join(FileProcessor.ALLOWED_EXTENSIONS)}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            cv_text = await CVParser.extract_text_from_file(temp_file_path)

            cv_data = await CVParser.parse_cv_data(cv_text)
            folder = f"user_{user_id}" if user_id else f"session_{session_id}"
            file.file.seek(0)
            file_path, file_name = await storage_service.upload_file(
                file.file,
                filename=file.filename,
                content_type=file.content_type,
                folder=folder
            )

            result = {
                "file_info": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "extension": file_extension,
                    "file_path": file_path
                },
                "cv_text": cv_text,
                "cv_data": cv_data,
                "personal_info": cv_data.get("personal_info", {})
            }

            return result

        finally:
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

    @staticmethod
    def get_file_extension(filename: str) -> str:
        return os.path.splitext(filename)[1].lower()

    @staticmethod
    def get_content_type(filename: str) -> str:
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or "application/octet-stream"
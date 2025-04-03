import uuid
import os
from typing import BinaryIO, Tuple, Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


class StorageService:
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=settings.R2_PUBLIC_ENDPOINT,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        )
        self.bucket_name = settings.R2_BUCKET_NAME

    async def upload_file(
            self,
            file: BinaryIO,
            filename: Optional[str] = None,
            content_type: Optional[str] = None,
            folder: str = "uploads"
    ) -> Tuple[str, str]:
        if not filename:
            filename = f"{uuid.uuid4()}.bin"

        file_uuid = uuid.uuid4()
        file_key = f"{folder}/{file_uuid}-{filename}"

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            try:
                self.client.upload_fileobj(
                    file,
                    self.bucket_name,
                    file_key,
                    ExtraArgs=extra_args
                )

                file_url = f"{settings.R2_PUBLIC_ENDPOINT}/{self.bucket_name}/{file_key}"
                return file_url, filename

            except ClientError as e:
                print(f"Error uploading file to R2: {e}")

                local_dir = f"/tmp/{folder}"
                os.makedirs(local_dir, exist_ok=True)

                local_path = f"{local_dir}/{file_uuid}-{filename}"

                file.seek(0)

                data = file.read()

                with open(local_path, 'wb') as f:
                    f.write(data)

                print(f"Saved file locally at: {local_path}")
                return f"local://{local_path}", filename

        except Exception as e:
            print(f"Error in upload_file: {e}")

            try:
                local_dir = f"/tmp/{folder}"
                os.makedirs(local_dir, exist_ok=True)

                local_path = f"{local_dir}/{file_uuid}-{filename}"

                file.seek(0)

                with open(local_path, 'wb') as f:
                    f.write(file.read())

                print(f"Fallback: Saved file locally at: {local_path}")
                return f"local://{local_path}", filename
            except Exception as inner_e:
                print(f"Failed even local fallback: {inner_e}")
                return f"placeholder://{folder}/{file_uuid}-{filename}", filename

    async def delete_file(self, file_key: str) -> bool:
        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            return True
        except ClientError as e:
            print(f"Error deleting file from R2: {e}")
            return False

    async def get_presigned_url(
            self,
            file_key: str,
            expiration: int = 3600
    ) -> Optional[str]:
        try:
            if file_key.startswith("local://") or file_key.startswith("placeholder://"):
                return file_key

            response = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_key,
                },
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None

storage_service = StorageService()
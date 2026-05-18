import io
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional

from minio import Minio

from app.shared.config import settings
from app.infrastructure.storage.base import BaseStorage


class MinioStorage(BaseStorage):

    def __init__(self, endpoint, access_key, secret_key, secure=False):
        # 初始化客户端（全局只需一次）
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = "tai-knowledge-base"
        self._ensure_bucket()

    def _ensure_bucket(self):
        """内部方法：确保桶存在"""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload(self, file: BinaryIO, object_name: str, bucket_name: Optional[str] = None):
        if isinstance(file, io.BytesIO):
            length = file.getbuffer().nbytes
        else:
            # 针对普通文件或其他流，回退到 fstat 或 seek/tell
            try:
                length = os.fstat(file.fileno()).st_size
            except (io.UnsupportedOperation, AttributeError):
                current_pos = file.tell()
                file.seek(0, os.SEEK_END)
                length = file.tell()
                file.seek(current_pos)

        return self.client.put_object(
            bucket_name=bucket_name or self.bucket_name,
            object_name=object_name,
            data=file,
            length=length,
            content_type="application/octet-stream"
        )

    def get_url(self, object_name, bucket_name: Optional[str] = None) -> str:
        bn = bucket_name or self.bucket_name
        return self.client.presigned_get_object(bn, object_name)

    def download_local(
            self,
            object_name: str,
            bucket_name: str = None
    ) -> str:

        bucket_name = bucket_name or self.bucket_name

        suffix = Path(object_name).suffix

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )

        response = self.client.get_object(
            bucket_name,
            object_name,
        )

        try:

            for data in response.stream(1024 * 1024):
                temp_file.write(data)

        finally:

            response.close()
            response.release_conn()
            temp_file.close()

        return temp_file.name


_minio_storage: MinioStorage | None = None


def get_minio_storage() -> MinioStorage:
    global _minio_storage
    if _minio_storage is None:
        _minio_storage = MinioStorage(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_id,
            secret_key=settings.minio.access_secret,
        )
    return _minio_storage

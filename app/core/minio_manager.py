import io
import os
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO, Optional

from minio import Minio
from fastapi import UploadFile

from app.core.config import settings
from app.core.logger import logger


class MinioManager:
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

    def get_presigned_url(self, object_name: str, expires_hours=1):
        return self.client.get_presigned_url(
            "POST",
            self.bucket_name,
            object_name,
            expires=timedelta(hours=expires_hours)
        )

    def upload(self, file: BinaryIO, object_name: str, bucket_name: Optional[str] = None):
        """上传文件并返回对象名称"""
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

    async def upload_file(self, file: UploadFile) -> str:
        """上传文件并返回对象名称"""
        file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
        object_name = f"{uuid.uuid4()}.{file_extension}"

        # 执行上传
        minio_res = self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=file.file,
            length=file.size,
            content_type=file.content_type
        )

        logger.info(f"Uploaded {object_name} successfully. {minio_res}")

        return object_name

    def get_url(self, object_name: str, expires_hours=1):
        """获取临时预览/下载地址"""
        return self.client.get_presigned_url(
            "GET",
            self.bucket_name,
            object_name,
            expires=timedelta(hours=expires_hours)
        )

    def download_to_temp(
            self,
            object_name: str,
            bucket_name: str = None
    ) -> str:
        """
        下载文件到临时目录
        :param object_name: 对象名称
        :param bucket_name: 桶名称
        :return: 临时文件路径
        """

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


# 实例化单例
minio_manager = MinioManager(
    endpoint=settings.minio.endpoint,
    access_key=settings.minio.access_id,
    secret_key=settings.minio.access_secret,
)


# 配合FastAPI依赖注入
def get_minio_manager():
    return minio_manager

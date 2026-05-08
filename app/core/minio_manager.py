import uuid
from datetime import timedelta
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


# 实例化单例
minio_manager = MinioManager(
    endpoint=settings.minio.endpoint,
    access_key=settings.minio.access_id,
    secret_key=settings.minio.access_secret,
)


# 配合FastAPI依赖注入
def get_minio_manager():
    return minio_manager

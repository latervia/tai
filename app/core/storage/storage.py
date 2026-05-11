from abc import ABC, abstractmethod
from typing import BinaryIO, Optional


class BaseStorage(ABC):

    @abstractmethod
    def upload(self, file: BinaryIO, object_name, bucket_name: Optional[str] = None) -> str:
        """
        上传文件并返回对象名称

        Args:
            file: 文件对象
            object_name: 对象名称
            bucket_name: 桶名称

        Returns:
        对象名称

        """
        pass

    @abstractmethod
    def get_url(self, object_name, bucket_name: Optional[str] = None) -> str:
        """
        获取文件url

        Args:
            object_name: 对象名称
            bucket_name: 桶名称

        Returns:
        文件url
        """
        pass

    @abstractmethod
    def download_local(self, object_name, bucket_name: Optional[str] = None) -> str:
        """
        下载文件到临时目录

        Args:
            object_name: 对象名称
            bucket_name: 桶名称

        Returns:
        临时文件路径
        """
        pass

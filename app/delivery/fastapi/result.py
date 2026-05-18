# 定义泛型变量
from typing import TypeVar, Generic, Optional, Any

from pydantic import BaseModel

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """通用返回体模型"""
    code: int = 0
    message: str = "success"
    data: Optional[T] = None

    @classmethod
    def success(cls, data: Any = None, message: str = "success"):
        return cls(code=0, message=message, data=data)

    @classmethod
    def fail(cls, code: int = 1, message: str = "fail", data: Any = None):
        return cls(code=code, message=message, data=data)

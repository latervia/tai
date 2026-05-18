from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    content: str
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryResult(Memory):
    score: float = 0.0


class BaseMemory(ABC):

    @abstractmethod
    async def save(self, session_id: str, memories: list[Memory]) -> None:
        """批量保存记忆"""
        ...

    @abstractmethod
    async def recall(self, session_id: str, query: str, *, top_k: int = 5) -> list[MemoryResult]:
        """检索相关记忆"""
        ...

    @abstractmethod
    async def forget(self, session_id: str) -> int:
        """清除指定 session 的全部记忆，返回清除条数"""
        ...

    @abstractmethod
    async def decay(self, session_id: str, older_than_days: int = 30) -> int:
        """衰减旧记忆，返回影响条数"""
        ...

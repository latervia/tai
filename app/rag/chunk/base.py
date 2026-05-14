from abc import ABC, abstractmethod

from app.rag.chunk.models import Chunk


class BaseChunker(ABC):

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        pass

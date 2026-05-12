# 知识块切分流程
from abc import ABC, abstractmethod


class BaseChunker(ABC):

    @abstractmethod
    def chunk(self, text: str):
        pass

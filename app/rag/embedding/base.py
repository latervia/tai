from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """将单条查询文本转换为向量"""
        pass

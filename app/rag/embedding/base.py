from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @abstractmethod
    def embed(self):
        pass
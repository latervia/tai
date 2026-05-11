from abc import ABC, abstractmethod


class ChunkStrategy(ABC):

    @abstractmethod
    def chunk(self, text):
        pass


class FAQChunkStrategy(ChunkStrategy):

    def chunk(self, text):
        pass

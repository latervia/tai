from app.rag.chunk.base import BaseChunker

"""
按照{"Q": "问题", "A": "答案"}的格式来切分文本
返回格式为[{"Q": "问题", "A": "答案"}]
"""

class AqChunker(BaseChunker):

    def chunk(self, text: str) -> list[str]:
        pass

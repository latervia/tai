import re

from app.domain.rag.chunk.base import BaseChunker
from app.domain.rag.chunk.models import Chunk


class QaChunker(BaseChunker):
    """按 Q-A 对格式切分文本，返回每个 Q-A 对作为一个 chunk。"""

    def __init__(self, qa_pattern: str | None = None):
        self.pattern = qa_pattern or r"(?:Q|问)[:：]\s*(.*?)\s*(?:A|答)[:：]\s*(.*?)(?=(?:Q|问)[:：]|\Z)"

    def chunk(self, text: str) -> list[Chunk]:
        pairs = re.findall(self.pattern, text, re.DOTALL | re.IGNORECASE)
        if not pairs:
            return [Chunk(content=text, chunk_index=0)]

        chunks = []
        for i, (question, answer) in enumerate(pairs):
            content = f"Q: {question.strip()}\nA: {answer.strip()}"
            chunks.append(Chunk(
                content=content,
                chunk_index=i,
                metadata={"question": question.strip(), "answer": answer.strip()},
            ))
        return chunks

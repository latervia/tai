from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.domain.rag.chunk.base import BaseChunker
from app.domain.rag.chunk.models import Chunk


class LangchainChunker(BaseChunker):
    HEADER_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    def __init__(self):
        super().__init__()
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADER_TO_SPLIT_ON,
            strip_headers=False,
        )

    def chunk(self, text: str) -> list[Chunk]:
        chunks = self.splitter.split_text(text)
        return [
            Chunk(
                content=chunk.page_content,
                chunk_index=i,
                metadata=chunk.metadata,
            )
            for i, chunk in enumerate(chunks)
        ]

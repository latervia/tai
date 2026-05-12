from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.rag.chunk.base import BaseChunker


class LangchainChunker(BaseChunker):
    """
    Langchain chunker
    """
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
        self.splitter = MarkdownHeaderTextSplitter(self.HEADER_TO_SPLIT_ON)

    def chunk(self, text: str):
        chunks = self.splitter.split_text(text)

        for chunk in chunks:
            print(chunk.metadata, chunk.page_content)

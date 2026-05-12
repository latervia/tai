from app.rag.parsers.base import BaseParser
from app.rag.parsers.docx_parser import DocxParser
from app.rag.parsers.fitz_parser import FitzParser


class ParserFactory:

    @staticmethod
    def create(
            source_type: str,
    ) -> BaseParser:

        if source_type == "pdf":
            return FitzParser()

        if source_type == "docx":
            return DocxParser()

        raise ValueError(
            f"Unsupported type: {source_type}"
        )

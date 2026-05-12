from app.rag.chunk.llm_chunker import LlmChunker
from app.rag.convertors.fitz_convertor import FitzConvertor


def build_pipeline(doc_id):
    # 构建知识库的Pipeline

    # 1. 将原始文档转换成Markdown
    convertor = FitzConvertor()
    md = convertor.convert(doc_id)

    # 2. 将Markdown文档转换成Chunks
    chunker = LlmChunker()
    chunks = chunker.chunk_with_llm(md)

    print(chunks)

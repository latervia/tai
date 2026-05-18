from app.shared.logger import logger
from app.domain.rag.chunk.base import BaseChunker
from app.domain.rag.chunk.llm_chunker import LlmChunker
from app.domain.rag.chunk.models import Chunk
from app.domain.rag.convert.base import BaseConvertor
from app.domain.rag.convert.fitz_convertor import FitzConvertor
from app.domain.rag.embedding.base import BaseEmbedding
from app.domain.rag.embedding.ollama_embedding import OllamaEmbedding
from app.domain.rag.embedding.vector_store import VectorStore


class Pipeline:

    def __init__(
            self,
            convertor: BaseConvertor,
            chunker: BaseChunker,
            embedding: BaseEmbedding | None = None,
            tenant_id: str = "default",
    ):
        self.convertor = convertor
        self.chunker = chunker
        self.embedding = embedding
        self.tenant_id = tenant_id

    def run(self, doc_id: str) -> list[Chunk]:
        logger.info(f"Pipeline 开始处理文档: {doc_id}")

        md = self.convertor.convert(doc_id)
        logger.info("文档已转换为 Markdown")

        chunks = self.chunker.chunk(md)
        for c in chunks:
            c.doc_id = doc_id
        logger.info(f"切分完成，共 {len(chunks)} 个 chunk")

        # chunk = Chunk("在许多应用中，可以通过标题和描述等丰富的信息集或文本、图像和音频等多种模式来搜索对象。", doc_id, 0,
        #               {"test": "test"})
        # chunks = [chunk]

        if self.embedding:
            store = VectorStore(self.embedding, tenant_id=self.tenant_id)
            store.index_chunks(chunks)
            logger.info(f"向量化完成，已入库 {len(chunks)} 条")

        return chunks


def build_pipeline(doc_id: str) -> list[Chunk]:
    pipeline = Pipeline(
        convertor=FitzConvertor(),
        chunker=LlmChunker(),
        embedding=OllamaEmbedding(),
    )
    return pipeline.run(doc_id)

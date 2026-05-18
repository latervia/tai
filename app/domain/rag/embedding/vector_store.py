from app.shared.logger import logger
from app.infrastructure.vectorstore.milvus import get_milvus_client
from app.domain.rag.chunk.models import Chunk
from app.domain.rag.embedding.base import BaseEmbedding


class VectorStore:

    def __init__(self, embedding: BaseEmbedding, tenant_id: str = "default"):
        self.embedding = embedding
        self.milvus = get_milvus_client()
        self.tenant_id = tenant_id

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[int]:
        all_ids = []
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            texts = [c.content for c in batch]
            vectors = self.embedding.embed(texts)

            ids = self.milvus.insert_vectors(
                doc_ids=[c.doc_id for c in batch],
                chunk_indices=[c.chunk_index for c in batch],
                documents=texts,
                dense_vectors=vectors,
                metadata_list=[c.metadata for c in batch],
                tenant_ids=[self.tenant_id] * len(batch),
            )
            all_ids.extend(ids)
            logger.info(f"已索引 {batch_start + len(batch)}/{len(chunks)} 个 chunk")

        return all_ids

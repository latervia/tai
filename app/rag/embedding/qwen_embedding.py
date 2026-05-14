from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.rag.embedding.base import BaseEmbedding


class QwenEmbedding(BaseEmbedding):

    def __init__(self, model: str = "text-embedding-v4", dim: int = 1024):
        self.client = OpenAIEmbeddings(
            api_key=settings.dashscope.api_key,
            base_url=settings.dashscope.base_url,
            model=model,
            dimensions=dim,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)

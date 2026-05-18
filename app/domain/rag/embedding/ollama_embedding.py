from langchain_ollama import OllamaEmbeddings

from app.shared.config import settings
from app.domain.rag.embedding.base import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):

    def __init__(self, model: str = "qwen3-embedding:0.6b", dim: int = 1024):
        self.client = OllamaEmbeddings(
            base_url=settings.ollama.base_url,
            model=model,
            dimensions=dim,
        )
        # self.client = OpenAIEmbeddings(
        #     api_key=settings.ollama.api_key,
        #     base_url=settings.ollama.base_url,
        #     model=model,
        #     dimensions=dim,
        # )

    def embed(self, texts: list[str]) -> list[list[float]]:
        print(f"长度: {len(texts)}")
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)

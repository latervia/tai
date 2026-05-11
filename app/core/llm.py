from functools import lru_cache

from langchain.chat_models.base import _ConfigurableModel, init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.core.config import settings


@lru_cache(maxsize=4)
def qwen(model: str = "qwen3.5-35b-a3b", temperature: float = 0.7) -> BaseChatModel | _ConfigurableModel:
    return init_chat_model(
        model=model,
        temperature=temperature,
        model_provider="openai",
        api_key=settings.dashscope.api_key,
        base_url=settings.dashscope.base_url,
    )


@lru_cache(maxsize=4)
def ollama() -> BaseChatModel:
    return ChatOllama(
        model="qwen3-vl:2b",
        base_url=settings.ollama.base_url,
    )

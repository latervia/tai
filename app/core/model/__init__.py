from app.core.model.model_factory import ModelFactory, ModelProvider
from app.core.model.model_dispatcher import ModelDispatcher, BudgetExceededError

# 便捷工厂函数 — 快速获取 LLM 实例
from app.core.config import settings


def qwen():
    return ModelFactory.get_model(ModelProvider.QWEN)


def ollama():
    return ModelFactory.get_model(ModelProvider.OLLAMA)


def ollama_vl():
    return ModelFactory.get_model(ModelProvider.OLLAMA, model=settings.ollama.vl_model)


__all__ = [
    "ModelFactory",
    "ModelProvider",
    "ModelDispatcher",
    "BudgetExceededError",
    "qwen",
    "ollama",
    "ollama_vl",
]

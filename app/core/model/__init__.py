from app.core.config import settings
from app.core.model.model_factory import ModelFactory, ModelProvider


def qwen():
    return ModelFactory.get_model(ModelProvider.QWEN)


def ollama():
    return ModelFactory.get_model(ModelProvider.OLLAMA)


def ollama_vl():
    return ModelFactory.get_model(ModelProvider.OLLAMA, model=settings.ollama.vl_model)

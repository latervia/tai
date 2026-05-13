from enum import Enum
from functools import lru_cache
from typing import Dict, Any

from langchain_openai import ChatOpenAI

from app.core.config import settings


class ModelProvider(Enum):
    QWEN = "qwen"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"


class ModelFactory:
    @staticmethod
    @lru_cache(maxsize=10)
    def get_model(provider: ModelProvider, **kwargs: Any) -> ChatOpenAI:
        """
        根据供应商获取配置好的 ChatOpenAI 实例
        kwargs 可以覆盖默认设置，如 temperature, model_name 等
        """
        configs: Dict[ModelProvider, Dict[str, Any]] = {
            ModelProvider.QWEN: {
                "openai_api_key": settings.dashscope.api_key,
                "openai_api_base": settings.dashscope.base_url,
                "model_name": "qwen-max",  # 默认模型
            },
            ModelProvider.OLLAMA: {
                "openai_api_key": "ollama",
                "openai_api_base": settings.ollama.base_url,
                "model_name": "llama3",
            },
            ModelProvider.DEEPSEEK: {
                "openai_api_key": settings.deepseek.api_key,
                "openai_api_base": "https://api.deepseek.com/v1",
                "model_name": "deepseek-chat",
            }
        }

        if provider not in configs:
            raise ValueError(f"Unsupported provider: {provider}")

        # 合并默认配置和用户传入的自定义参数
        final_config = {**configs[provider], **kwargs}

        return ChatOpenAI(**final_config)
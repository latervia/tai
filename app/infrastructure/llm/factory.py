import posixpath
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI

from app.shared.config import settings


class ModelProvider(Enum):
    QWEN = "qwen"
    OLLAMA = "ollama"


class ModelFactory:
    @staticmethod
    @lru_cache(maxsize=10)
    def get_model(provider: ModelProvider, model: Optional[str] = None, **kwargs: Any) -> ChatOpenAI:
        """
        根据供应商获取配置好的 ChatOpenAI 实例
        kwargs 可以覆盖默认设置，如 temperature, model_name 等
        """
        configs: Dict[ModelProvider, Dict[str, Any]] = {
            ModelProvider.QWEN: {
                "api_key": settings.dashscope.api_key,
                "base_url": settings.dashscope.base_url,
                "model": settings.dashscope.model,  # 默认模型
            },
            ModelProvider.OLLAMA: {
                "api_key": settings.ollama.api_key,
                "base_url": posixpath.join(settings.ollama.base_url, "v1"),
                "model": settings.ollama.model,
            }
        }

        if provider not in configs:
            raise ValueError(f"Unsupported provider: {provider}")

        # 1. 提取基础配置
        final_config = configs[provider].copy()

        # 2. 如果显式传入了 model，则覆盖基础配置中的 model
        if model:
            final_config["model"] = model

        # 3. 最后合并 kwargs（可以覆盖 temperature, base_url 等，甚至再次覆盖 model）
        final_config.update(kwargs)

        print(final_config)

        return ChatOpenAI(**final_config)

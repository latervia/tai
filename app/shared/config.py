from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """基础通用配置域"""

    model_config = SettingsConfigDict(
        env_file=".env.tailscale",
        # env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True
    )


class DashScopeSettings(BaseConfig):
    """阿里百炼配置域"""

    api_key: str = Field(alias="DASHSCOPE_API_KEY")
    model: str = Field(alias="DASHSCOPE_MODEL")
    base_url: str = Field(alias="DASHSCOPE_BASE_URL")
    temperature: float = Field(alias="DASHSCOPE_TEMPERATURE")


class OllamaSettings(BaseConfig):
    """Ollama配置域"""

    model: str = Field(alias="OLLAMA_MODEL")
    vl_model: str = Field(alias="OLLAMA_VL_MODEL")
    embedding_model: str = Field(alias="OLLAMA_EMBEDDING_MODEL")
    api_key: str = Field(alias="OLLAMA_API_KEY")
    base_url: str = Field(alias="OLLAMA_BASE_URL")
    temperature: float = Field(alias="OLLAMA_TEMPERATURE")


class MilvusSettings(BaseConfig):
    """Milvus配置域"""

    host: str = Field(alias="MILVUS_HOST")
    port: int = Field(alias="MILVUS_PORT")
    user: str = Field(alias="MILVUS_USER")
    password: str = Field(alias="MILVUS_PASSWORD")
    database: str = Field(alias="MILVUS_DATABASE")
    collection: str = Field(alias="MILVUS_COLLECTION")
    dim: int = Field(default=1024, alias="MILVUS_DIM")


class PostgresSettings(BaseConfig):
    """Postgres配置域"""
    debug: bool = Field(alias="POSTGRES_DEBUG")
    url: str = Field(alias="POSTGRES_URL")


class MinioSettings(BaseConfig):
    """Minio配置域"""

    bucket: str = Field(alias="MINIO_BUCKET")
    access_id: str = Field(alias="MINIO_ACCESS_ID")
    access_secret: str = Field(alias="MINIO_ACCESS_SECRET")
    endpoint: str = Field(alias="MINIO_ENDPOINT")

class RedisSettings(BaseConfig):
    host: str = Field(alias="REDIS_HOST")
    port: int = Field(alias="REDIS_PORT")
    db: int = Field(alias="REDIS_DB")
    password: str = Field(alias="REDIS_PASSWORD")


class Settings(BaseConfig):
    """全局配置"""

    dashscope: DashScopeSettings = Field(default_factory=DashScopeSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)


@lru_cache
def get_settings() -> Settings:
    """单例配置(避免重复加载 .env)"""
    return Settings()


settings = get_settings()

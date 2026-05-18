import json
from typing import Any, Optional, Union

import redis.asyncio as redis
from redis.asyncio.client import Redis

from app.shared.config import settings
from app.shared.logger import logger


class RedisManager:
    """
    Redis 客户端管理类（单例）
    封装了连接池管理与常用操作，支持自动 JSON 序列化
    """
    _instance: Optional["RedisManager"] = None
    _client: Optional[Redis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def init_redis(
            self,
            host: str = settings.redis.host,
            port: int = settings.redis.port,
            db: int = settings.redis.db,
            password: Optional[str] = settings.redis.password
    ):
        """初始化异步连接池"""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,  # 自动将字节转为字符串
                    max_connections=20,  # 连接池上限
                    socket_timeout=5.0
                )
                # 检查连接
                await self._client.ping()
                logger.info(f"Redis 成功连接至 {host}:{port}/{db}")
            except Exception as e:
                logger.error(f"Redis 连接失败: {e}")
                raise e

    async def close(self):
        """关闭连接池"""
        if self._client:
            await self._client.aclose()
            logger.info("Redis 连接已安全关闭")

    @property
    def connection(self) -> Redis:
        """获取底层 Redis 实例"""
        if self._client is None:
            raise RuntimeError("Redis 未初始化，请先调用 init_redis()")
        return self._client

    # --- 高级封装方法 ---

    async def set_object(self, key: str, value: Any, ex: int = None):
        """序列化存储对象 (Dict, List 等)"""
        val = json.dumps(value, ensure_ascii=False)
        await self._client.set(key, val, ex=ex)

    async def get_object(self, key: str) -> Optional[Union[dict, list]]:
        """获取并反序列化对象"""
        data = await self._client.get(key)
        return json.loads(data) if data else None

    async def delete(self, key: str):
        """删除键"""
        await self._client.delete(key)


# 导出单例对象
redis_client = RedisManager()

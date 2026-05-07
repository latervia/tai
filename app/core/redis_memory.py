from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver

from app.core.config import settings

redis_checkpointer = RedisSaver.from_conn_string(
    redis_url=f"redis://{settings.redis.host}:{settings.redis.port}"
)

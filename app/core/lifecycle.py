# 如果有 Milvus / Redis / ES 也在这里初始化
from app.core.logger import logger
from app.core.redis_manager import redis_client


async def startup_event():
    logger.info("Starting up...")
    # await redis_client.init_redis()
    # await init_llm()
    # init_vector_db()
    # init_cache()


async def shutdown_event():
    logger.info("Shutting down...")
    # await close_llm()

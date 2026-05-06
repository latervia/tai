# 如果有 Milvus / Redis / ES 也在这里初始化

async def startup_event():
    print("Starting up...")
    # await init_llm()
    # init_vector_db()
    # init_cache()


async def shutdown_event():
    print("Shutting down...")
    # await close_llm()
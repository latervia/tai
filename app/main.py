from fastapi import FastAPI
from contextlib import asynccontextmanager

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import chat, rag
from app.core.lifecycle import startup_event, shutdown_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段（加载模型 / 初始化向量库）
    await startup_event()
    yield
    # 关闭阶段（释放资源）
    await shutdown_event()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agent System",
        version="1.0.0",
        description="Multi-Agent + RAG System",
        lifespan=lifespan,
        openapi_prefix="/api"
    )

    # 注册路由
    app.include_router(chat.router)
    app.include_router(rag.router)

    # 健康检查
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()


# 中间件（日志 + tracing）
@app.middleware("http")
async def log_middleware(request, call_next):
    response = await call_next(request)
    return response


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 异常统一处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )


if __name__ == '__main__':
    import uvicorn

    # 这里开启 reload=True 可能会导致部分调试器二次进入断点，
    # 调试复杂问题时建议设为 False
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
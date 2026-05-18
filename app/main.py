"""向后兼容：FastAPI app 已迁移到 app.delivery.fastapi。

新代码请使用:
    from app.delivery.fastapi import app
    # 或
    uvicorn app.delivery.fastapi.main:app
"""

from app.delivery.fastapi.main import app, create_app

__all__ = ["app", "create_app"]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.delivery.fastapi.main:app", host="0.0.0.0", port=8000, reload=True)

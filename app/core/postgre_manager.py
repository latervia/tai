from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(settings.postgres.url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Dependency: 用于 FastAPI 自动管理 Session 开启与关闭
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

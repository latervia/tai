from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.postgre_manager import get_db
from app.models.postgre_models import KBModel
from app.schemas.rag import KbCreateReq


class RagService:

    def __init__(self, db: Session):
        self.db = db

    def create(self, req: KbCreateReq):
        existing = self.db.execute(select(KBModel.name == req.name)).scalar_one_or_none()

        if existing:
            raise ValueError(f"知识库{req.name}已存在")

        new_kb = KBModel(
            name=req.name,
            description=req.description
        )

        self.db.add(new_kb)
        self.db.commit()
        self.db.refresh(new_kb)
        return new_kb


def get_rag_service(db: Session = Depends(get_db)):
    return RagService(db)

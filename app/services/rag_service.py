from uuid import UUID

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.postgre_manager import get_db
from app.models.postgre_models import KBModel, DocumentModel
from app.schemas.rag import KBCreateReq


class RagService:

    def __init__(self, db: Session):
        self.db = db

    def create(self, req: KBCreateReq) -> KBModel:
        existing = self.db.execute(select(KBModel.name == req.name)).scalar_one_or_none()

        if existing:
            raise ValueError(f"知识库{req.name}已存在")

        new_kb = KBModel(
            name=req.name,
            description=req.description,
            milvus_collection="tai"
        )

        self.db.add(new_kb)
        self.db.commit()
        self.db.refresh(new_kb)
        return new_kb

    def list(self):
        return self.db.execute(select(KBModel)).scalars().all()

    def delete(self, kb_id: int):
        self.db.execute(select(KBModel).where(KBModel.id == kb_id))
        self.db.commit()

    def doc_list(self, kb_id: UUID):
        return self.db.execute(select(DocumentModel).where(DocumentModel.kb_id == kb_id)).scalars().all()


def get_rag_service(db: Session = Depends(get_db)):
    return RagService(db)

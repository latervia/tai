from uuid import UUID

from fastapi import Depends, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.core.minio_manager import MinioManager, get_minio_manager
from app.core.postgre_manager import get_db
from app.models.postgre_models import KBModel, DocumentModel
from app.schemas.rag import KBCreateReq


class RagService:

    def __init__(self, db: Session, minio: MinioManager):
        self.db = db
        self.minio = minio

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
        if not kb_id:
            return self.db.execute(select(DocumentModel)).scalars().all()
        else:
            return self.db.execute(select(DocumentModel)).scalars().all()

    async def doc_upload(self, file: UploadFile):
        file_name = await self.minio.upload_file(file) # todo 添加一个回调任务调用doc_convertor
        logger.info(f"文件上传成功，文件名：{file_name}")
        # 存储到数据库
        self.db.add(DocumentModel(file_name=str(file_name)))
        self.db.commit()


def get_rag_service(db: Session = Depends(get_db), minio: MinioManager = Depends(get_minio_manager)):
    return RagService(db, minio)

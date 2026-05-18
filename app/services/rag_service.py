from uuid import UUID

from fastapi import Depends, UploadFile, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.shared.logger import logger
from app.infrastructure.database.postgres import get_db
from app.infrastructure.storage.minio import MinioStorage, get_minio_storage
from app.infrastructure.database.models import KBModel, DocumentModel
from app.domain.rag.pipeline.build_pipeline import build_pipeline
from app.delivery.fastapi.schemas.rag import KBCreateReq


class RagService:

    def __init__(self, db: Session, minio: MinioStorage):
        self.db = db
        self.minio = minio
        # self.parser = Parser(minio)
        # self.convertor: BaseConvertor = FitzConvertor()

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

    def doc_list(self, kb_id: UUID | None):
        if not kb_id:
            return self.db.execute(select(DocumentModel)).scalars().all()
        else:
            return self.db.execute(select(DocumentModel)).scalars().all()

    async def doc_upload(self, file: UploadFile, bg_tasks: BackgroundTasks):

        doc_id = str(uuid7())
        doc_name = f"{doc_id}/source.pdf"
        print(f"上传文件名称: {doc_name}")

        self.minio.upload(file.file, object_name=doc_name)  # todo 添加一个回调任务调用doc_convertor
        logger.info("文件上传成功")
        # 存储到数据库
        self.db.add(DocumentModel(file_name=str(doc_id)))
        self.db.commit()

        # 开启后台线程调用文档解析
        # bg_tasks.add_task(self.convertor.convert, doc_id)
        bg_tasks.add_task(build_pipeline, doc_id)


def get_rag_service(
        db: Session = Depends(get_db),
        minio: MinioStorage = Depends(get_minio_storage)
) -> RagService:
    return RagService(db, minio)

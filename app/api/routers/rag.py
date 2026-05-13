from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, BackgroundTasks

from app.api.rest.result import Result
from app.schemas.rag import KBCreateReq, KBRes, DocumentRes
from app.services.rag_service import RagService, get_rag_service

router = APIRouter(prefix="/rag")


@router.post("/create", description="创建知识库")
def knowledge_base_create(req: KBCreateReq, service: RagService = Depends(get_rag_service)):
    res = service.create(req)
    return Result.success(data=res)


@router.post("/list", response_model=Result[List[KBRes]], description="获取知识库列表")
def knowledge_base_list(service: RagService = Depends(get_rag_service)):
    res = service.list()
    return Result.success(data=res)


@router.post("/delete/{kb_id}", description="删除知识库")
def knowledge_base_delete(kb_id: int, service: RagService = Depends(get_rag_service)):
    res = service.delete(kb_id)
    return Result.success(data=res)


@router.post("/doc/list", response_model=Result[List[DocumentRes]], description="获取知识库文档列表")
def knowledge_base_doc_list(kb_id: UUID = None, service: RagService = Depends(get_rag_service)):
    res = service.doc_list(kb_id)
    return Result.success(data=res)


@router.post("/doc/upload", description="上传文档")
async def knowledge_base_doc_upload(
        file: UploadFile,
        bg_tasks: BackgroundTasks,
        service: RagService = Depends(get_rag_service)
):
    await service.doc_upload(file, bg_tasks)
    return Result.success()


@router.post("/doc/bind", description="关联文档到知识库")
def knowledge_base_doc_bind():
    pass


@router.post("/doc/unbind", description="取消关联文档到知识库")
def knowledge_base_doc_unbind():
    pass

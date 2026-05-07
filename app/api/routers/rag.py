from fastapi import APIRouter, Depends

from app.services.rag_service import RagService, get_rag_service

router = APIRouter(prefix="/rag")


@router.post("/list", description="获取知识库列表")
def knowledge_base_list():
    pass


@router.post("/create", description="创建知识库")
def knowledge_base_create(service: RagService = Depends(get_rag_service)):
    pass


@router.post("/delete", description="删除知识库")
def knowledge_base_delete():
    pass


@router.post("/doc/upload", description="上传文档")
def knowledge_base_doc_upload():
    pass


@router.post("/doc/bind", description="关联文档到知识库")
def knowledge_base_doc_bind():
    pass


@router.post("/doc/unbind", description="取消关联文档到知识库")
def knowledge_base_doc_unbind():
    pass

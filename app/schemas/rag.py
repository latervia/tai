from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class KBCreateReq(BaseModel):
    name: str = Field(..., description="知识库名称")
    description: str = Field(..., description="知识库描述")


class KBRes(BaseModel):
    id: UUID
    name: str
    description: str

    model_config = ConfigDict(from_attributes=True)


class DocumentRes(BaseModel):
    id: UUID
    kb_id: UUID
    file_name: str
    file_type: str
    file_size: int
    doc_hash: str
    source_url: str
    status: str
    error_msg: str

    model_config = ConfigDict(from_attributes=True)

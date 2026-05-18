from typing import Optional
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
    # kb_id: UUID
    file_name: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    doc_hash: Optional[str] = None
    source_url: Optional[str] = None
    status: Optional[str] = None
    error_msg: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

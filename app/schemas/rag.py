from pydantic import BaseModel, Field


class KbCreateReq(BaseModel):
    name: str = Field(..., description="知识库名称")
    description: str = Field(..., description="知识库描述")

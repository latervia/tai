from typing import List, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Literal["human", "ai", "system"]
    content: str


class ChatReq(BaseModel):
    session_id: str = Field(..., description="Session ID")
    parent_id: str = Field(..., description="Parent message ID")
    message: str = Field(..., description="Human message")

    # 扩展参数
    temperature: float = 0.7

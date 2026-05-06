from typing import List

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str


class ChatReq(BaseModel):
    session_id: str = Field(..., description="Session ID")
    messages: List[Message]

    # 扩展参数
    temperature: float = 0.7

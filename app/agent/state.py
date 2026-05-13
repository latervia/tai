from typing import TypedDict, List, Annotated, NotRequired, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class State(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], add_messages]  # 对话消息
    summary_memory: NotRequired[str]  # 历史摘要
    current_intent: NotRequired[Optional[str]]  # 当前意图

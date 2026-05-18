from typing import TypedDict, Annotated, NotRequired, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from app.domain.agentx.states.trace import Trace


class State(TypedDict, total=False):
    # 对话状态
    session_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    intent: NotRequired[str]  # 当前意图

    # Trace 追踪
    trace: NotRequired[List[Trace]]

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

    # 研究管线
    research_plan: NotRequired[dict]  # Planner 产出的大纲
    search_results: NotRequired[list]  # Search Worker 的搜索结果
    collected_sources: NotRequired[list]  # Read Worker 抓取的原文
    extracted_facts: NotRequired[dict]  # Extract Worker 产出的结构化素材
    draft_report: NotRequired[dict]  # Writer 产出的初稿，含 content + citations
    review_result: NotRequired[dict]  # Reviewer 的评审裁决，含 verdict + issues

    # Trace 追踪
    trace: NotRequired[List[Trace]]

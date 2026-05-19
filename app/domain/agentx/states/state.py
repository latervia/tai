from typing import TypedDict, Annotated, NotRequired, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from app.domain.agentx.states.trace import Trace


# ---------------------------------------------------------------
# 列表累加 reducer — 用于跨循环合并 search_results / collected_sources
# ---------------------------------------------------------------
def _merge_list(left: list | None, right: list | None) -> list:
    """合并两个列表，dict 项按 url 去重"""
    result: list = []
    seen: set[str] = set()

    for item in (left or []):
        if isinstance(item, dict) and item.get("url"):
            seen.add(item["url"])
        result.append(item)

    for item in (right or []):
        if isinstance(item, dict):
            url = item.get("url", "")
            if not url or url not in seen:
                if url:
                    seen.add(url)
                result.append(item)
        else:
            result.append(item)

    return result


class State(TypedDict, total=False):
    # 对话状态
    session_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    intent: NotRequired[str]  # 当前意图

    # 研究管线
    research_plan: NotRequired[dict]  # Planner 产出的大纲（一次写入，不覆盖）
    search_results: Annotated[list, _merge_list]  # 累加：每次补充搜索的结果
    collected_sources: Annotated[list, _merge_list]  # 累加：每次抓取的原文
    extracted_facts: NotRequired[dict]  # 覆盖：每次根据全部 sources 重新萃取
    draft_report: NotRequired[dict]  # 覆盖：每次重写
    review_result: NotRequired[dict]  # 覆盖：每次重审
    research_loop_count: NotRequired[int]  # 回退次数计数器

    # Trace 追踪
    trace: NotRequired[List[Trace]]

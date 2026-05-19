import json

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.agentx.states.state import State
from app.domain.agentx.workers.chat_worker import ChatWorker
from app.domain.agentx.workers.research.planner import Planner
from app.domain.agentx.workers.research.search_worker import SearchWorker
from app.domain.agentx.workers.research.read_worker import ReadWorker
from app.domain.agentx.workers.research.extract_worker import ExtractWorker
from app.domain.agentx.workers.research.write_worker import WriteWorker
from app.domain.agentx.workers.research.review_worker import ReviewWorker

MAX_RESEARCH_LOOPS = 3

_INTENT_PROMPT = """\
分析用户意图，仅输出 JSON：

{"intent": "chat"}      — 闲聊、简单问答、问候
{"intent": "research"}  — 要求撰写报告、深度调研、多源综合分析

严格只输出 JSON，不要有任何其他文字。"""


def build_graph(model):
    """构建多场景 Agent 图。

    路由结构：
      START → 意图分类 ─┬─ chat     → ChatWorker → END
                        └─ research → Planner → Search → Read → Extract
                                        ↑                       │
                                        │    ┌──────────────────┘
                                        │    ▼
                                        │  Write → Review ─┬─ pass → END
                                        │                   └─ revise → 回 Search
                                        └──────────────────────┘
    """

    # -- Worker 实例 --
    chat = ChatWorker(model)
    planner = Planner(model)
    search = SearchWorker(model)
    read = ReadWorker(model)
    extract = ExtractWorker(model)
    write = WriteWorker(model)
    review = ReviewWorker(model)

    graph = StateGraph(State)

    # -- 节点 --
    graph.add_node("_classify", _make_classifier(model))
    graph.add_node("chat_worker", chat)

    graph.add_node("planner", planner)
    graph.add_node("search", search)
    graph.add_node("read", read)
    graph.add_node("extract", extract)
    graph.add_node("write", write)
    graph.add_node("review", review)
    graph.add_node("_increment_loop", _increment_loop)

    # -- 顶层路由 --
    graph.add_edge(START, "_classify")
    graph.add_conditional_edges(
        "_classify",
        _route_by_intent,
        {
            "chat": "chat_worker",
            "research": "planner",
        },
    )
    graph.add_edge("chat_worker", END)

    # -- Research 管线 --
    graph.add_edge("planner", "search")
    graph.add_edge("search", "read")
    graph.add_edge("read", "extract")
    graph.add_edge("extract", "write")
    graph.add_edge("write", "review")

    graph.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "pass": END,
            "revise": "_increment_loop",
            "force_pass": END,
        },
    )
    graph.add_edge("_increment_loop", "search")

    return graph.compile()


# ================================================================
# 辅助函数
# ================================================================

def _make_classifier(model):
    """返回意图分类节点函数（闭包捕获 model）"""

    async def _classify_intent(state: State) -> dict:
        # 已有意图则保持不变（避免回退时重复分类）
        if state.get("intent"):
            return {}

        messages = state.get("messages", [])
        if not messages:
            return {"intent": "chat"}

        last_msg = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

        response = await model.ainvoke([
            SystemMessage(content=_INTENT_PROMPT),
            HumanMessage(content=last_msg),
        ])

        try:
            text = response.content.strip()
            # 容错：提取可能的 JSON 块
            if "{" in text:
                text = text[text.index("{"):text.rindex("}") + 1]
            result = json.loads(text)
            intent = result.get("intent", "chat")
            return {"intent": intent if intent in ("chat", "research") else "chat"}
        except (json.JSONDecodeError, ValueError, AttributeError):
            return {"intent": "chat"}

    return _classify_intent


def _route_by_intent(state: State) -> str:
    intent = state.get("intent", "chat")
    if intent == "research":
        return "research"
    return "chat"


def _route_after_review(state: State) -> str:
    review_result = state.get("review_result", {})
    verdict = review_result.get("verdict", "pass")
    loop_count = state.get("research_loop_count", 0)

    if verdict == "pass":
        return "pass"

    if loop_count >= MAX_RESEARCH_LOOPS:
        return "force_pass"

    return "revise"


async def _increment_loop(state: State) -> dict:
    count = state.get("research_loop_count", 0)
    return {"research_loop_count": count + 1}

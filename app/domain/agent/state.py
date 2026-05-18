"""多 Agent 系统的共享 State 定义"""
from typing import TypedDict, Annotated, NotRequired, Optional, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class MultiAgentState(TypedDict, total=False):
    """多 Agent 共享状态 — LangGraph 的共享"黑板"

    total=False 使所有字段变为可选，各节点只需返回自己关心的字段。
    """

    # ── 对话基础（继承自原 State） ────────────────────────
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    summary_memory: NotRequired[str]

    # ── 多 Agent 编排 ─────────────────────────────────────
    current_agent: NotRequired[str]
    # 上一步负责处理的 Agent 名称（Supervisor 用此决策下一步）

    agent_outputs: NotRequired[dict[str, Any]]
    # 各 Agent 的输出产物，key 为 agent name
    # 例如: {"rag_agent": {"docs": [...], "status": "done"}, "chat_agent": {...}}

    next_agent: NotRequired[Optional[str]]
    # Supervisor 决策的下一步 Agent 名称，None 表示直接结束

    # ── 任务规划 ──────────────────────────────────────────
    task_plan: NotRequired[list[dict]]
    # 当前任务的分解步骤，例如:
    # [{"step":1, "agent":"rag_agent", "goal":"检索相关文档", "status":"pending"}, ...]

    # ── 人机协作 ──────────────────────────────────────────
    pending_approval: NotRequired[Optional[dict]]
    # 等待人类审批的操作信息
    # {"tool": "delete_records", "args": {...}, "reason": "用户要求删除"}

    # ── 可观测性 ──────────────────────────────────────────
    trace: NotRequired[list[dict]]
    # 本次请求的完整追踪事件列表

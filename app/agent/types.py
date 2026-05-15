"""多 Agent 系统的共享类型定义"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class AgentStatus(str, Enum):
    """Agent 执行状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    DONE = "done"
    FAILED = "failed"


class ThoughtType(str, Enum):
    """Agent 思考结果的类型"""
    RESPOND = "respond"        # 直接回复用户
    CALL_TOOL = "call_tool"    # 需要调用工具
    DELEGATE = "delegate"      # 委托给其他 Agent
    ASK_HUMAN = "ask_human"    # 需要人类介入


@dataclass
class ToolDef:
    """工具定义 — 描述一个可被 Agent 调用的工具"""
    name: str
    description: str                          # 给 LLM 看的功能描述
    fn: Callable                              # 实际执行的函数
    parameters: Optional[dict] = None         # JSON Schema 参数定义
    permissions: list[str] = field(default_factory=list)  # 所需权限，如 ["read_file"]
    requires_approval: bool = False           # 是否需要人类审批


@dataclass
class AgentThought:
    """Agent 思考后的决策"""
    type: ThoughtType
    content: str                              # 回复内容 或 工具调用说明
    tool_name: Optional[str] = None           # CALL_TOOL 时的工具名
    tool_args: Optional[dict] = None          # CALL_TOOL 时的参数
    target_agent: Optional[str] = None        # DELEGATE 时的目标 Agent 名
    confidence: float = 0.0                   # 置信度 0~1


@dataclass
class ActionResult:
    """工具执行或 Agent 行动的结果"""
    success: bool
    output: Any = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class TraceEvent:
    """可观测性追踪事件"""
    agent_name: str
    event_type: str                           # "llm_call" | "tool_call" | "think" | "error"
    timestamp: float
    input: Any = None
    output: Any = None
    tokens: Optional[dict] = None             # {"input": N, "output": M}
    duration_ms: float = 0.0
    error: Optional[str] = None

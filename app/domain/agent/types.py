"""多 Agent 系统的共享类型定义"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    DONE = "done"
    FAILED = "failed"


class RiskLevel(str, Enum):
    """操作风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolDef:
    """工具定义"""
    name: str
    description: str
    fn: Callable
    parameters: Optional[dict] = None               # JSON Schema 参数定义
    permissions: list[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW            # 高于 MEDIUM 需审批


@dataclass
class ActionResult:
    """工具执行结果"""
    success: bool
    output: Any = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
    duration_ms: float = 0.0
    needs_approval: bool = False                     # 需要人类审批才能继续


@dataclass
class ApprovalRequest:
    """等待审批的操作"""
    request_id: str
    tool_name: str
    tool_args: dict
    reason: str
    risk_level: RiskLevel = RiskLevel.MEDIUM
    status: str = "pending"

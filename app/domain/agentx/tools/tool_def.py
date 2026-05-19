from dataclasses import dataclass, field
from typing import Callable, Optional

from app.domain.agentx.workers.base import RiskLevel


@dataclass
class ToolDef:
    """工具定义"""
    name: str
    description: str
    fn: Callable
    parameters: Optional[dict] = None  # JSON Schema 参数定义
    permissions: list[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW  # 高于 MEDIUM 需审批

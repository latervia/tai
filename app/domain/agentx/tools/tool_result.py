from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    output: Any = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
    duration_ms: float = 0.0
    needs_approval: bool = False  # 需要人类审批才能继续

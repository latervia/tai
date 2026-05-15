# 向后兼容：工具模块已迁移到 app.agent.tools
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.message_adapter import MessageAdapter

__all__ = ["ToolRegistry", "MessageAdapter"]

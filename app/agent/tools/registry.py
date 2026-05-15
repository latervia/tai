"""工具注册表 — 统一管理 Agent 可调用的工具，支持权限过滤"""
from typing import Callable, Optional

from app.agent.types import ToolDef


class ToolRegistry:
    """全局工具注册中心

    新工具只需注册一次，各 Agent 通过权限过滤获取可用工具集。
    """
    _tools: dict[str, ToolDef] = {}
    # 每个 Agent 被授予的权限列表
    _agent_permissions: dict[str, list[str]] = {}

    # ── 注册 ──────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        fn: Callable,
        *,
        description: str = "",
        parameters: Optional[dict] = None,
        permissions: Optional[list[str]] = None,
        requires_approval: bool = False,
    ):
        """注册一个工具到全局注册表"""
        cls._tools[name] = ToolDef(
            name=name,
            description=description,
            fn=fn,
            parameters=parameters,
            permissions=permissions or [],
            requires_approval=requires_approval,
        )

    # ── 授权 ──────────────────────────────────────────────

    @classmethod
    def grant(cls, agent_name: str, permissions: list[str]):
        """授予某个 Agent 一组工具权限"""
        cls._agent_permissions[agent_name] = permissions

    # ── 查询 ──────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> Optional[ToolDef]:
        """按名称获取工具定义"""
        return cls._tools.get(name)

    @classmethod
    def get_for_agent(cls, agent_name: str) -> list[ToolDef]:
        """获取某 Agent 有权使用的工具列表"""
        allowed = cls._agent_permissions.get(agent_name, [])
        return [
            tool for tool in cls._tools.values()
            if any(p in allowed for p in tool.permissions)
        ]

    @classmethod
    def list_all(cls) -> dict[str, ToolDef]:
        """返回所有已注册工具（调试用）"""
        return dict(cls._tools)

    @classmethod
    def clear(cls):
        """清空注册表（测试用）"""
        cls._tools.clear()
        cls._agent_permissions.clear()

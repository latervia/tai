"""工具注册表 — 统一管理 Agent 可调用的工具，支持权限过滤"""
from typing import Callable, Optional

from app.agent.types import ToolDef, RiskLevel


class ToolRegistry:
    """全局工具注册中心

    新工具只需注册一次，各 Agent 通过权限过滤获取可用工具集。
    """
    _tools: dict[str, ToolDef] = {}
    _agent_permissions: dict[str, list[str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        fn: Callable,
        *,
        description: str = "",
        parameters: Optional[dict] = None,
        permissions: Optional[list[str]] = None,
        risk_level: RiskLevel = RiskLevel.LOW,
    ):
        cls._tools[name] = ToolDef(
            name=name,
            description=description,
            fn=fn,
            parameters=parameters,
            permissions=permissions or [],
            risk_level=risk_level,
        )

    @classmethod
    def grant(cls, agent_name: str, permissions: list[str]):
        """授予某个 Agent 一组工具权限"""
        cls._agent_permissions[agent_name] = permissions

    @classmethod
    def get(cls, name: str) -> Optional[ToolDef]:
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
        return dict(cls._tools)

    @classmethod
    def clear(cls):
        cls._tools.clear()
        cls._agent_permissions.clear()

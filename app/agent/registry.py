"""Agent 注册表 — 集中管理所有可用 Agent 的元数据"""
from typing import Optional

from app.agent.base import BaseAgent


class AgentRegistry:
    """Agent 注册中心

    供 Supervisor 查询可用 Agent 列表，决定路由目标。
    每个注册的 Agent 包含类引用和元数据。
    """
    _entries: dict[str, dict] = {}

    @classmethod
    def register(cls, agent_cls: type[BaseAgent]):
        """注册一个 Agent 类"""
        name = agent_cls.AGENT_NAME
        cls._entries[name] = {
            "name": name,
            "description": agent_cls.AGENT_DESCRIPTION,
            "cls": agent_cls,
        }

    @classmethod
    def list_agents(cls) -> list[dict]:
        """返回所有已注册 Agent 的元数据列表"""
        return [
            {"name": e["name"], "description": e["description"]}
            for e in cls._entries.values()
        ]

    @classmethod
    def get(cls, name: str) -> Optional[dict]:
        """按名称获取 Agent 注册信息"""
        return cls._entries.get(name)

    @classmethod
    def get_cls(cls, name: str) -> Optional[type[BaseAgent]]:
        """按名称获取 Agent 类"""
        entry = cls._entries.get(name)
        return entry["cls"] if entry else None

    @classmethod
    def clear(cls):
        """清空注册表（测试用）"""
        cls._entries.clear()
